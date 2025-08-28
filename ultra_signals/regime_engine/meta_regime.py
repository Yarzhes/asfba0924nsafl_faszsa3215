"""Sprint 43 Meta-Regime Engine (Advanced Implementation)

Provides a pluggable `RegimeModelBundle` that fuses multi-family features to
produce probabilistic market regimes plus ancillary risk context:
  - regime_probs (soft distribution over configured regimes)
  - regime_label (argmax / heuristic mapping)
  - transition_hazard (flip risk proxy from HMM or entropy / change-point)
  - exp_vol_h (HAR‑lite expected realized volatility)
  - dir_bias (short-horizon directional drift −1..+1)
  - policy gates (size multiplier, veto counts, downgrade flag)

Design principles:
  * All heavy dependencies optional (hdbscan, hmmlearn, matplotlib)
  * Defensive fallbacks when models not yet trained
  * Lazy incremental fit triggered on cadence or buffer milestones
  * Persistence with feature column hash for future compatibility
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math, hashlib, time
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import joblib

from ultra_signals.core.custom_types import FeatureVector, RegimeFeatures
from ultra_signals.core.config import RegimeSettings
from .feature_builder import FeatureAssembler
from .clusterer import adaptive_cluster, map_clusters_to_regimes
from .policy_loader import load_policy_map

try:  # Optional HMM
    from hmmlearn.hmm import GaussianHMM
    _HAVE_HMM = True
except Exception:  # pragma: no cover
    _HAVE_HMM = False


class CusumDetector:
    """Simple two-sided CUSUM for detecting abrupt confidence shifts."""
    def __init__(self, k: float = 0.5, h: float = 4.0):
        self.k = k; self.h = h
        self.pos = 0.0; self.neg = 0.0
        self.last_signal = 0
    def update(self, x: float) -> int:
        self.pos = max(0.0, self.pos + x - self.k)
        self.neg = min(0.0, self.neg + x + self.k)
        signal = 0
        if self.pos > self.h:
            signal = 1; self.pos = 0.0; self.neg = 0.0
        elif self.neg < -self.h:
            signal = -1; self.pos = 0.0; self.neg = 0.0
        self.last_signal = signal
        return signal


@dataclass
class VolForecasterHAR:
    """HAR‑lite realized volatility forecaster.

    Keeps a rolling buffer of realized vol (or proxy). Forecast is weighted
    average of short / medium / long windows.
    """
    horizon: int = 12
    window: int = 400
    rv_buffer: List[float] = field(default_factory=list)
    def push(self, rv: float):
        if rv is not None and not math.isnan(rv):
            self.rv_buffer.append(float(rv))
            if len(self.rv_buffer) > self.window:
                self.rv_buffer.pop(0)
    def forecast(self) -> Optional[float]:
        if len(self.rv_buffer) < 30:
            return None
        arr = np.array(self.rv_buffer[-self.horizon*5:])
        if arr.size == 0:
            return None
        s = arr[-self.horizon:].mean()
        m = arr[-min(len(arr), self.horizon*2):].mean()
        l = arr.mean()
        return float(0.6*s + 0.3*m + 0.1*l)


@dataclass
class SupervisedLayer:
    """Lightweight calibrated logistic layer (optional)."""
    enabled: bool
    model: Any | None = None
    calibrator: Any | None = None
    last_fit_ts: int = 0
    def fit(self, X: np.ndarray, y: np.ndarray, calibrate: bool = True):
        if not self.enabled or X.shape[0] < 50:
            return
        self.model = LogisticRegression(max_iter=250)
        self.model.fit(X, y)
        if calibrate:
            try:
                prob = self.model.predict_proba(X)[:,1]
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(prob, y)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Calibration failed: {e}")
        self.last_fit_ts = int(time.time())
    def predict(self, X: np.ndarray) -> Optional[float]:
        if not self.enabled or self.model is None or X.shape[0]==0:
            return None
        prob = self.model.predict_proba(X)[:,1][-1]
        if self.calibrator is not None:
            try:
                prob = float(self.calibrator.transform([prob])[0])
            except Exception:  # pragma: no cover
                pass
        return float(prob)


@dataclass
class RegimeModelBundle:
    settings: RegimeSettings
    assembler: FeatureAssembler = field(default_factory=FeatureAssembler)
    regimes: List[str] = field(default_factory=list)
    policy_map: Dict[str, Dict[str, Any]] | None = None
    cluster_labels: List[int] = field(default_factory=list)
    cluster_mapping: Dict[int,str] = field(default_factory=dict)
    hmm: Any | None = None
    supervised: SupervisedLayer | None = None
    cusum: CusumDetector = field(default_factory=CusumDetector)
    vol_forecaster: VolForecasterHAR = field(default_factory=VolForecasterHAR)
    sticky_probs: Dict[str,float] = field(default_factory=dict)
    last_regime: Optional[str] = None
    flip_count: int = 0
    bar_count: int = 0
    purity_history: List[float] = field(default_factory=list)
    downgrade_flag: bool = False
    feature_columns_hash: Optional[str] = None
    version: str = "0.2"

    def __post_init__(self):
        if not self.regimes:
            self.regimes = self.settings.regimes
        self.policy_map = load_policy_map(self.settings.policy_map_path)
        sup_cfg = self.settings.supervised or {}
        self.supervised = SupervisedLayer(enabled=bool(sup_cfg.get('enabled', False)))

    # ---------------------- MODEL REFRESH ----------------------
    def _refresh_models(self):
        X, cols = self.assembler.matrix()
        if X.shape[0] < 50:  # not enough data
            return
        try:
            labels, _meta = adaptive_cluster(X, self.settings.model_dump())
            self.cluster_labels = list(labels)
            self.cluster_mapping = map_clusters_to_regimes(X, labels, self.regimes)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Clustering failed: {e}")
        # Optional HMM
        hmm_cfg = self.settings.hmm or {}
        if hmm_cfg.get('enabled', True) and _HAVE_HMM and len(set(self.cluster_labels)) > 1:
            try:
                n_states = int(hmm_cfg.get('n_states', len(set([l for l in self.cluster_labels if l!=-1]))))
                self.hmm = GaussianHMM(n_components=n_states, covariance_type=hmm_cfg.get('covariance_type','diag'), n_iter=50)
                self.hmm.fit(X)
            except Exception as e:  # pragma: no cover
                logger.warning(f"HMM fit failed: {e}")
                self.hmm = None
        # Supervised layer (vol spike heuristic target)
        if self.supervised and self.supervised.enabled:
            try:
                vol_proxy = np.abs(X[:,0])
                thr = np.percentile(vol_proxy, 75)
                y = (vol_proxy >= thr).astype(int)
                self.supervised.fit(X, y)
            except Exception as e:  # pragma: no cover
                logger.debug(f"Supervised fit failed: {e}")
        # Purity metric
        try:
            total_var = float(np.var(X, axis=0).mean())
            intra = 0.0; counts = 0
            labels_arr = np.array(self.cluster_labels)
            for c in set(labels_arr):
                if c == -1: continue
                seg = X[labels_arr==c]
                if seg.shape[0] < 2: continue
                intra += float(np.var(seg, axis=0).mean()) * seg.shape[0]
                counts += seg.shape[0]
            purity = 1.0 - (intra / (counts * total_var) if counts and total_var>0 else 0.0)
            self.purity_history.append(purity)
            self._update_downgrade(purity)
        except Exception:  # pragma: no cover
            pass
        self.feature_columns_hash = hashlib.sha256("|".join(cols).encode()).hexdigest()

    def _update_downgrade(self, purity: float):
        if purity < 0.25 and len(self.purity_history) > 50:
            self.downgrade_flag = True
        if self.bar_count > 50 and self.flip_count / max(1,self.bar_count) > 0.30:
            self.downgrade_flag = True

    # ---------------------- PERSISTENCE ----------------------
    def save(self, path: str):
        state = {
            'version': self.version,
            'regimes': self.regimes,
            'cluster_mapping': self.cluster_mapping,
            'policy_map': self.policy_map,
            'feature_hash': self.feature_columns_hash,
            'purity_history': self.purity_history,
            'sticky_probs': self.sticky_probs,
        }
        try:
            joblib.dump(state, path)
            logger.info(f"Regime bundle saved to {path}")
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed saving regime bundle: {e}")

    def load(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        try:
            state = joblib.load(path)
            self.regimes = state.get('regimes', self.regimes)
            self.cluster_mapping = state.get('cluster_mapping', {})
            self.policy_map = state.get('policy_map', self.policy_map)
            self.feature_columns_hash = state.get('feature_hash')
            self.purity_history = state.get('purity_history', [])
            self.sticky_probs = state.get('sticky_probs', {})
            logger.info(f"Loaded regime bundle from {path}")
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed loading regime bundle: {e}")

    # ---------------------- PUBLIC FIT TRIGGER ----------------------
    def fit_if_due(self):
        if self.bar_count % max(1, int(self.settings.retrain_interval_hours*60)) == 0 or not self.cluster_mapping:
            self._refresh_models()

    # ---------------------- INFERENCE ----------------------
    def infer(self, fv: FeatureVector) -> RegimeFeatures:
        base = fv.regime or RegimeFeatures()
        self.assembler.push(fv)
        self.bar_count += 1
        if self.bar_count % 50 == 0 and not self.downgrade_flag:
            self._refresh_models()
        X, _cols = self.assembler.matrix()
        probs = {r: 1.0/len(self.regimes) for r in self.regimes}
        regime_label = None
        hazard = 0.0
        pre_smoothed = None
        # Cluster assignment (nearest centroid)
        if self.cluster_labels:
            try:
                last_vec = X[-1]
                labels_arr = np.array(self.cluster_labels)
                centroids = {c: X[labels_arr==c].mean(axis=0) for c in set(labels_arr) if c != -1}
                if centroids:
                    dists = {c: float(np.linalg.norm(last_vec - v)) for c,v in centroids.items()}
                    best_cluster = min(dists, key=dists.get)
                    regime_label = self.cluster_mapping.get(best_cluster)
            except Exception:  # pragma: no cover
                pass
        # Fallback heuristics using legacy regime.mode
        if regime_label is None and base.mode.value == 'trend' and 'trend_up' in self.regimes:
            regime_label = 'trend_up'
        if regime_label is None and base.mode.value == 'chop' and 'chop_lowvol' in self.regimes:
            regime_label = 'chop_lowvol'
        if regime_label:
            probs = {r: 0.0001 for r in self.regimes}
            probs[regime_label] = 0.6
        # Supervised adjustment
        if self.supervised:
            sup_prob = self.supervised.predict(X)
            if sup_prob is not None and regime_label:
                probs[regime_label] = min(1.0, max(0.0, 0.5*probs[regime_label] + 0.5*sup_prob))
        # Capture pre-smoothed copy for diagnostics
        try:
            pre_smoothed = probs.copy()
        except Exception:
            pre_smoothed = None
        # Sticky smoothing
        stick = float((self.settings.smoothing or {}).get('stickiness', 0.85))
        if self.sticky_probs and regime_label:
            smoothed = {}
            for r in self.regimes:
                prev = self.sticky_probs.get(r, 0.0)
                raw = probs.get(r, 0.0)
                smoothed[r] = stick*prev + (1-stick)*raw
            probs = smoothed
        # Moving-average smoothing over last N bars if configured
        ma_window = int((self.settings.smoothing or {}).get('ma_window_bars', 1))
        if ma_window and ma_window > 1:
            # maintain a tiny history buffer on the bundle
            hist = getattr(self, '_prob_hist', [])
            hist.append(probs.copy())
            if len(hist) > ma_window:
                hist.pop(0)
            self._prob_hist = hist
            # average across history
            avg = {r: 0.0 for r in self.regimes}
            for row in hist:
                for r,v in row.items():
                    avg[r] += float(v)
            denom = max(1, len(hist))
            probs = {r: avg[r]/denom for r in avg}
        smoothed_probs = probs.copy()
        self.sticky_probs = probs.copy()
        # Normalize
        s = sum(probs.values())
        if s > 0:
            probs = {k: v/s for k,v in probs.items()}
        # Hazard from HMM else entropy
        if self.hmm and regime_label and regime_label in self.regimes:
            try:
                trans = self.hmm.transmat_
                hazard = 1 - float(trans[0,0])  # naive (state 0 self-transition complement)
            except Exception:  # pragma: no cover
                pass
        else:
            entropy = -sum(p*math.log(p+1e-9) for p in probs.values())
            max_e = math.log(len(probs)) if probs else 1.0
            hazard = entropy / max_e if max_e>0 else 0.0
        # Compute entropy & max_prob for diagnostics
        try:
            entropy = -sum(p*math.log(p+1e-9) for p in probs.values())
            max_e = math.log(len(probs)) if probs else 1.0
            norm_entropy = entropy / max_e if max_e>0 else 0.0
            max_prob = max(probs.values()) if probs else 0.0
        except Exception:
            norm_entropy = 1.0
            max_prob = 0.0
        # Change-point (confidence deltas)
        delta_conf = (base.confidence or 0.0) - (getattr(self, '_prev_conf', 0.0))
        self._prev_conf = base.confidence or 0.0
        if self.cusum.update(delta_conf) != 0:
            hazard = min(1.0, max(hazard, 0.75))
        # Expected vol: push atr_percentile as crude proxy
        self.vol_forecaster.push(base.atr_percentile or 0.0)
        exp_vol = self.vol_forecaster.forecast()
        dir_bias = None
        if fv.trend and fv.trend.ema_medium and fv.trend.ema_long:
            diff = fv.trend.ema_medium - fv.trend.ema_long
            scale = abs(fv.trend.ema_long) or 1.0
            dir_bias = max(-1.0, min(1.0, diff/scale))
        if regime_label and self.last_regime and regime_label != self.last_regime:
            self.flip_count += 1
        if regime_label:
            self.last_regime = regime_label
        # Policy application
        applied_policy = self.policy_map.get(regime_label, {}) if regime_label else {}
        size_mult = applied_policy.get('size_mult', 1.0)
        if self.downgrade_flag:
            size_mult *= 0.5
        rf = base.model_copy(deep=True)
        rf.regime_probs = probs
        rf.pre_smoothed_regime_probs = pre_smoothed
        rf.smoothed_regime_probs = smoothed_probs
        rf.regime_label = regime_label
        rf.transition_hazard = round(hazard,4)
        rf.regime_trans_prob = round(hazard,4)
        rf.exp_vol_h = exp_vol
        rf.dir_bias = dir_bias
        rf.regime_entropy = round(norm_entropy,4)
        rf.regime_max_prob = round(max_prob,4)
        # Confidence flag from config bands
        bands = getattr(self.settings, 'confidence_bands', {'high':0.7,'medium':0.4})
        if max_prob >= float(bands.get('high',0.7)):
            cf = 'high'
        elif max_prob >= float(bands.get('medium',0.4)):
            cf = 'medium'
        else:
            cf = 'low'
        rf.regime_confidence_flag = cf
        # Suggest a playbook hint based on regime_label and confidence
        try:
            if regime_label:
                hint = f"{regime_label}"
                # low confidence -> conservative suffix
                if cf == 'low':
                    hint = f"{hint}-conservative"
                rf.playbook_hint = hint
        except Exception:
            rf.playbook_hint = None
        # adjusted confidence blending vol forecast
        try:
            vol_adj = 1.0
            if exp_vol is not None:
                # higher expected vol reduces confidence
                vol_adj = max(0.2, 1.0 - 0.5 * float(exp_vol))
            rf.regime_confidence_adj = round(max_prob * vol_adj,4)
        except Exception:
            # best-effort, do not fail inference on failure here
            rf.regime_confidence_adj = None
        # Inject context (macro / whale)
        if fv.macro and fv.macro.macro_risk_regime:
            rf.macro_risk_context = fv.macro.macro_risk_regime
        if fv.macro and fv.macro.macro_extreme_flag is not None:
            rf.sent_extreme_flag = fv.macro.macro_extreme_flag
        whales_attr = getattr(fv, 'whales', None)
        if whales_attr is not None:
            pressure = getattr(whales_attr, 'composite_pressure_score', None)
            if pressure is not None:
                rf.whale_pressure = pressure
        rf.gates = rf.gates or {}
        rf.gates.update({
            'regime_size_mult': size_mult,
            'regime_veto_count': len(applied_policy.get('veto', [])),
            'downgrade': self.downgrade_flag
        })
        return rf

    # ---------------------- DIAGNOSTICS ----------------------
    def export_transition_matrix(self, path: str):  # pragma: no cover
        if not self.hmm:
            return
        try:
            import csv
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                for row in self.hmm.transmat_:
                    w.writerow([f"{x:.5f}" for x in row])
        except Exception as e:
            logger.error(f"Failed exporting transition matrix: {e}")


# Convenience singleton
_BUNDLE: Optional[RegimeModelBundle] = None

def get_regime_bundle(settings: RegimeSettings) -> RegimeModelBundle:
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = RegimeModelBundle(settings=settings)
    return _BUNDLE

__all__ = ["RegimeModelBundle", "get_regime_bundle"]
