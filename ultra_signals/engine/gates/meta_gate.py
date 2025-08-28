"""Meta Probability Gate (Sprint 31)

Applies a calibrated win-probability model to candidate trades and maps the
probability to ENTER | DAMPEN | VETO with regime/profile specific thresholds.

Lightweight: avoids importing heavy ML libs at import time (lazy load model).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
from loguru import logger
import time

try:  # optional heavy imports inside functions
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


@dataclass(slots=True)
class MetaGateDecision:
    action: str               # ENTER | DAMPEN | VETO
    p: Optional[float] = None # calibrated probability
    threshold: Optional[float] = None
    profile: Optional[str] = None
    reason: Optional[str] = None
    size_mult: Optional[float] = None
    widen_stop_mult: Optional[float] = None
    meta: Dict[str, Any] | None = None


class MetaGate:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings or {}
        self._model_cache: Any | None = None
        self._feature_names: list[str] | None = None
        self._last_load_mtime: float | None = None

    # ---------------- Internal helpers -----------------
    def _cfg(self) -> Dict[str, Any]:
        return (self.settings.get("meta_scorer") or {}) if isinstance(self.settings, dict) else {}

    def _ensure_model(self):
        cfg = self._cfg()
        if not cfg.get("enabled", True):
            return None
        model_path = cfg.get("model_path")
        if not model_path:
            return None
        # Lazy reload if file changed (mtime)
        try:
            import os
            mtime = os.path.getmtime(model_path)
            if self._model_cache is None or (self._last_load_mtime and mtime > self._last_load_mtime):
                if joblib is None:  # pragma: no cover
                    logger.warning("joblib not available; meta gate disabled")
                    return None
                self._model_cache = joblib.load(model_path)
                self._last_load_mtime = mtime
                # Expect dict with keys: model, feature_names, pre (optional)
                if isinstance(self._model_cache, dict) and 'feature_names' in self._model_cache:
                    self._feature_names = list(self._model_cache['feature_names'])
                # If model bundle didn't include feature_names, allow runtime configuration
                if not self._feature_names:
                    cfg_feats = cfg.get('input_features') or []
                    if cfg_feats:
                        try:
                            self._feature_names = list(cfg_feats)
                            logger.info(f"[MetaGate] No feature_names in bundle; using {len(self._feature_names)} features from settings.input_features")
                        except Exception:
                            self._feature_names = []
                else:
                    logger.info(f"[MetaGate] Loaded model from {model_path} with {len(self._feature_names or [])} features")
        except FileNotFoundError:
            logger.debug("Meta model path not found: {}", model_path)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed loading meta model: {e}")
        return self._model_cache

    def _predict(self, features: Dict[str, Any]) -> Optional[float]:
        mdl = self._ensure_model()
        if mdl is None:
            return None
        try:
            fnames = self._feature_names or []
            row = []
            for name in fnames:
                v = features.get(name)
                if v is None:
                    # simple safe default
                    v = 0.0
                row.append(float(v))
            import numpy as np
            X = np.asarray(row, dtype=float).reshape(1, -1)
            if 'pre' in mdl and mdl['pre'] is not None:
                X = mdl['pre'].transform(X)
            model = mdl.get('model') if isinstance(mdl, dict) else mdl
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0, 1]
            elif hasattr(model, 'predict'):
                pred = model.predict(X)[0]
                proba = float(pred)
            else:
                return None
            # Optional calibration wrapper
            if isinstance(mdl, dict) and mdl.get('calibrator') is not None:
                try:
                    cal = mdl['calibrator']
                    if hasattr(cal, 'predict_proba'):
                        proba = cal.predict_proba([[proba]])[0, 1] if cal.classes_.tolist() == [0,1] else proba
                except Exception:
                    pass
            return float(proba)
        except Exception as e:  # pragma: no cover
            logger.debug(f"MetaGate predict error: {e}")
            return None

    # ---------------- Public API -----------------
    def evaluate(self, side: str, regime_profile: str, feature_bundle: Dict[str, Any]) -> MetaGateDecision:
        cfg = self._cfg()
        if not cfg.get('enabled', True):
            return MetaGateDecision(action='ENTER', reason='META_DISABLED')
        side_u = (side or '').upper()
        if side_u not in ('LONG','SHORT'):
            return MetaGateDecision(action='ENTER', reason='NON_DIRECTIONAL')
        missing_policy = str(cfg.get('missing_policy','SAFE')).upper()
        p = self._predict(feature_bundle)
        if p is None:
            if missing_policy == 'OFF':
                return MetaGateDecision(action='ENTER', reason='MISSING_MODEL')
            if missing_policy == 'OPEN':
                return MetaGateDecision(action='ENTER', reason='MISSING_MODEL_OPEN')
            # SAFE
            return MetaGateDecision(action='DAMPEN', reason='MISSING_MODEL_SAFE', size_mult=cfg.get('fallback_size_mult', 0.6))

        # Threshold selection by profile
        thresholds = (cfg.get('thresholds') or {})
        partial_band = (cfg.get('partial_band') or {})
        prof = regime_profile or 'trend'
        th = float((thresholds.get(prof) if isinstance(thresholds.get(prof), (int,float)) else thresholds.get(prof, 0.55)) or 0.55)
        band_cfg = partial_band.get(prof) or {}
        low = float(band_cfg.get('low', th - 0.06))
        high = float(band_cfg.get('high', th))
        size_mult = band_cfg.get('size_mult')
        widen_mult = band_cfg.get('widen_stop_mult')

        action = 'ENTER'
        reason = 'META_PASS'
        if p < low:
            action = 'VETO'
            reason = 'META_LOW_PROB'
        elif p < high:
            action = 'DAMPEN'
            reason = 'META_PARTIAL'
        return MetaGateDecision(action=action, p=round(p,6), threshold=th, profile=prof, reason=reason, size_mult=size_mult if action=='DAMPEN' else None, widen_stop_mult=widen_mult if action=='DAMPEN' else None, meta={'low': low, 'high': high})


def evaluate_gate(side: str, regime_profile: str, feature_bundle: Dict[str, Any], settings: Dict[str, Any]):
    gate = MetaGate(settings)
    try:
        return gate.evaluate(side, regime_profile, feature_bundle)
    except Exception as e:  # pragma: no cover
        logger.debug(f"MetaGate error: {e}")
        return MetaGateDecision(action='ENTER', reason='ERR')

__all__ = ['MetaGate', 'MetaGateDecision', 'evaluate_gate']