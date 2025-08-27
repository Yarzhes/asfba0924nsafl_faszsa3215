"""Behavioral Finance Veto / Sizing Engine (Sprint 45)

MVP scaffolding that derives human-behavior risk context from existing
feature store data (price/OHLCV, flow_metrics, sentiment snapshot, whale
snapshot, macro) without any paid data sources.

Responsibilities (initial slice):
  * Compute rolling simple z-scores (using incremental mean/std estimator
    if historical window available) for returns, volume, OI rate, funding.
  * Build composite FOMO / Euphoria / Capitulation scores from available
    proxies (weights configurable).
  * Maintain basic session & time-of-day expectancy tables (in-memory hash
    updated every call; persistence hook for future version).
  * Apply rule-based policy -> behavior_action (ENTER/DAMPEN/VETO) + size multiplier.

Design notes:
  * All calculations are defensive: missing inputs -> skip component.
  * Hysteresis: once a FOMO/Euphoria/Capitulation regime asserted it requires
    the score to retrace below (threshold - hysteresis_delta) before clearing.
  * Expectancy: simple running average PnL surrogate via provided outcome
    accumulator hook (placeholder) -> for now we just compute frequency share
    per hour/session to get a pseudo quality percentile.

Public API:
  BehaviorEngine(settings, feature_store)
    .evaluate(symbol, ts_sec, feature_bundle) -> BehaviorFeatures

`feature_bundle` expected minimal keys:
  {
    'ohlcv': {'close': float, 'open': float, 'high': float, 'low': float, 'volume': float},
    'flow_metrics': FlowMetricsFeatures | None,
    'regime': RegimeFeatures | None,
    'sentiment': dict | None (snapshot),
    'whales': dict | None (snapshot)
  }

Future extensions:
  * Hawkes/Poisson burst detection for liquidations & social bursts
  * Persistent expectancy tables (sqlite / parquet)
  * Calibrated classifier for probabilistic behavior_veto_prob
"""
from __future__ import annotations

import math
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import datetime as dt
from pathlib import Path
import yaml
import pandas as pd

from loguru import logger

from ultra_signals.core.custom_types import BehaviorFeatures


@dataclass
class _RollStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squares of differences

    def update(self, x: float):  # Welford
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def std(self) -> Optional[float]:
        if self.n < 2:
            return None
        return math.sqrt(self.m2 / (self.n - 1))

    def z(self, x: float) -> Optional[float]:
        s = self.std()
        if s is None or s == 0:
            return None
        return (x - self.mean) / s


class BehaviorEngine:
    def __init__(self, settings: Dict[str, Any], feature_store: Any | None = None):
        self.settings = settings or {}
        beh_cfg = (self.settings.get('behavior') or {}) if isinstance(self.settings, dict) else {}
        self.enabled = bool(beh_cfg.get('enabled', True))
        self._fs = feature_store
        # Rolling stats caches keyed by symbol
        self._ret_stats: Dict[str, _RollStats] = {}
        self._vol_stats: Dict[str, _RollStats] = {}
        self._oi_rate_stats: Dict[str, _RollStats] = {}
        self._funding_stats: Dict[str, _RollStats] = {}
        # Hysteresis state per symbol
        self._state: Dict[str, str] = {}  # 'fomo'|'euphoria'|'capitulation'|''
        # Expectancy tallies
        self._hour_tally: Dict[int, int] = {}
        self._session_tally: Dict[str, int] = {}
        self._total_observations: int = 0
        # Config knobs
        self._thr = beh_cfg.get('thresholds', {
            'fomo_z': 1.5,
            'euphoria_z': 2.0,
            'capitulation_z': 2.0,
            'hysteresis_delta': 0.4,
        })
        self._weights = beh_cfg.get('weights', {
            'fomo': {'ret_z': 0.4, 'volume_z': 0.2, 'funding_z': 0.2, 'oi_rate_z': 0.2},
            'euphoria': {'funding_z': 0.35, 'ret_z': 0.25, 'sent_z': 0.25, 'whale_inflow': 0.15},
            'capitulation': {'ret_z': 0.35, 'volume_z': 0.2, 'liq_notional_z': 0.25, 'sent_z': 0.2},
        })
        self._policy = beh_cfg.get('policy', {
            'veto': {'euphoria_z': 2.5, 'fomo_z': 2.2, 'capitulation_z': 2.4},
            'dampen': {'fomo_z': 1.8, 'euphoria_z': 2.0, 'capitulation_z': 2.0, 'size_mult': 0.6},
            'base_size_mult': 1.0,
            'weekend_penalty': 0.85,
            'holiday_penalty': 0.75,
        })
        # Persistence paths
        self._storage_dir = Path(beh_cfg.get('storage_dir', '.cache/behavior'))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._expectancy_path = self._storage_dir / 'expectancy.parquet'
        self._holiday_path = Path(beh_cfg.get('holiday_calendar_path', 'holidays.yaml'))
        self._holidays = self._load_holidays(self._holiday_path)
        self._load_expectancy()
        # Calibration harness (collect labeled outcomes if provided externally)
        self._calib_samples: list[tuple[float,int]] = []  # (raw_score, outcome_veto)
        self._prob_model = None  # placeholder for logistic reg or isotonic calibrator

    # ------------------------------------------------------------------
    def evaluate(self, symbol: str, ts_sec: int, bundle: Dict[str, Any]) -> Optional[BehaviorFeatures]:
        if not self.enabled:
            return None
        try:
            ohlcv = bundle.get('ohlcv') or {}
            close_px = float(ohlcv.get('close')) if ohlcv.get('close') is not None else None
            open_px = float(ohlcv.get('open')) if ohlcv.get('open') is not None else None
            volume = float(ohlcv.get('volume')) if ohlcv.get('volume') is not None else None
        except Exception:
            close_px = open_px = volume = None
        # Return early if missing price
        if close_px is None or open_px is None:
            # -------------------- Core Data Extraction --------------------
            if not self.enabled:
                return None
            try:
                ohlcv = bundle.get('ohlcv') or {}
                close_px = float(ohlcv.get('close')) if ohlcv.get('close') is not None else None
                open_px = float(ohlcv.get('open')) if ohlcv.get('open') is not None else None
                volume = float(ohlcv.get('volume')) if ohlcv.get('volume') is not None else None
            except Exception:
                close_px = open_px = volume = None
            if close_px is None or open_px is None:
                return None
            # Return
            ret = None
            try:
                if open_px and open_px != 0:
                    ret = (close_px - open_px) / open_px
            except Exception:
                ret = None
            ret_z = self._update_and_z(self._ret_stats, symbol, ret) if ret is not None else None
            vol_z = self._update_and_z(self._vol_stats, symbol, volume) if volume is not None else None
            fm = bundle.get('flow_metrics')
            oi_rate = getattr(fm, 'oi_rate', None) if fm else None
            oi_rate_z = self._update_and_z(self._oi_rate_stats, symbol, oi_rate) if oi_rate is not None else None
            funding = None
            try:
                deriv = bundle.get('derivatives')
                if deriv:
                    funding = getattr(deriv, 'funding_now', None)
            except Exception:
                pass
            funding_z = self._update_and_z(self._funding_stats, symbol, funding) if funding is not None else None
            liq_cluster = getattr(fm, 'liq_cluster', None) if fm else None
            liq_notional = getattr(fm, 'liq_notional_sum', None) if fm else None
            liq_notional_z = None
            if liq_notional is not None and self._vol_stats.get(symbol) and self._vol_stats[symbol].std():
                try:
                    mu = self._vol_stats[symbol].mean
                    sd = self._vol_stats[symbol].std() or 1.0
                    liq_notional_z = (liq_notional - mu) / sd
                except Exception:
                    liq_notional_z = None
            sent = bundle.get('sentiment') or {}
            sent_z = sent.get('sent_z_s') if isinstance(sent, dict) else None
            whales = bundle.get('whales') or {}
            whale_inflow_z = whales.get('whale_inflow_z_s') or whales.get('whale_inflow_z_m') if isinstance(whales, dict) else None
            whale_withdrawal_flag = whales.get('exch_withdrawal_burst_flag') if isinstance(whales, dict) else None
            wick_body_ratio = None
            try:
                high = float(ohlcv.get('high')); low = float(ohlcv.get('low'))
                body = abs(close_px - open_px)
                rng = high - low
                if body > 0 and rng > 0:
                    wick_body_ratio = (rng - body) / body
            except Exception:
                pass
            # -------------------- Calendar / Session ----------------------
            dt_utc = dt.datetime.utcfromtimestamp(ts_sec)
            hour = dt_utc.hour
            session = 'asia' if 0 <= hour < 8 else 'eu' if 8 <= hour < 16 else 'us'
            dow = dt_utc.weekday()
            is_weekend = 1 if dow >= 5 else 0
            date_key = dt_utc.date().isoformat()
            is_holiday = 1 if date_key in self._holidays else 0
            self._hour_tally[hour] = self._hour_tally.get(hour, 0) + 1
            self._session_tally[session] = self._session_tally.get(session, 0) + 1
            self._total_observations += 1
            tod_expectancy = self._hour_tally[hour] / max(1, self._total_observations)
            session_expectancy = self._session_tally[session] / max(1, self._total_observations)
            try:
                counts = list(self._hour_tally.values())
                my = self._hour_tally[hour]
                below = sum(1 for c in counts if c <= my)
                tod_quality = below / max(1, len(counts))
            except Exception:
                tod_quality = None
            # -------------------- Scores & States -------------------------
            fomo_score = self._weighted_sum({'ret_z': ret_z, 'volume_z': vol_z, 'funding_z': funding_z, 'oi_rate_z': oi_rate_z}, self._weights['fomo'])
            euphoria_score = self._weighted_sum({'funding_z': funding_z, 'ret_z': ret_z, 'sent_z': sent_z, 'whale_inflow': whale_inflow_z}, self._weights['euphoria'])
            capitulation_score = self._weighted_sum({'ret_z': (-ret_z if ret_z is not None else None), 'volume_z': vol_z, 'liq_notional_z': liq_notional_z, 'sent_z': (-sent_z if sent_z is not None else None)}, self._weights['capitulation'])
            divergence_flag = self._detect_divergence(ret, fm)
            oi_purge_flag = 1 if (oi_rate_z is not None and oi_rate_z <= -2.0 and (vol_z or 0) > 1.0) else 0
            state = self._state.get(symbol, '')
            hyst = float(self._thr.get('hysteresis_delta', 0.4))
            state = self._update_state(state, fomo_score, euphoria_score, capitulation_score, hyst)
            self._state[symbol] = state
            action, size_mult, reason, flags = self._policy_decision(fomo_score, euphoria_score, capitulation_score, is_weekend, is_holiday)
            raw_score = max([x for x in [fomo_score, euphoria_score, capitulation_score] if x is not None] or [0.0])
            behavior_veto_prob = self._estimate_prob(raw_score)
            weekend_penalty = self._policy.get('weekend_penalty') if is_weekend else None
            holiday_penalty = self._policy.get('holiday_penalty') if is_holiday else None
            feat = BehaviorFeatures(
                beh_fomo_score_z=fomo_score,
                beh_euphoria_score_z=euphoria_score,
                beh_capitulation_score_z=capitulation_score,
                ret_z=ret_z,
                volume_z=vol_z,
                oi_rate_z=oi_rate_z,
                funding_z=funding_z,
                liq_cluster=liq_cluster,
                liq_notional_z=liq_notional_z,
                wick_body_ratio=wick_body_ratio,
                whale_net_inflow_z=whale_inflow_z,
                whale_withdrawal_flag=whale_withdrawal_flag,
                sent_z_s=sent_z,
                session=session,
                hour_bin=hour,
                dow=dow,
                is_weekend=is_weekend,
                is_holiday=is_holiday,
                session_expectancy=session_expectancy,
                tod_expectancy=tod_expectancy,
                tod_quality=tod_quality,
                weekend_penalty=weekend_penalty,
                holiday_penalty=holiday_penalty,
                behavior_veto=1 if action == 'VETO' else 0,
                behavior_action=action,
                behavior_size_mult=size_mult,
                behavior_reason=reason,
                flags=flags,
                hysteresis_state=state,
                behavior_veto_prob=behavior_veto_prob,
                divergence_flag=divergence_flag,
                oi_purge_flag=oi_purge_flag,
            )
            if self._total_observations % 500 == 0:
                self._save_expectancy()
            return feat
    def _update_and_z(self, cache: Dict[str, _RollStats], symbol: str, x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        rs = cache.get(symbol)
        if rs is None:
            rs = _RollStats()
            cache[symbol] = rs
        rs.update(x)
        return rs.z(x)

    def _weighted_sum(self, values: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
        num = 0.0; den = 0.0
        for k,w in weights.items():
            v = values.get(k)
            if v is None:
                continue
            num += w * v
            den += abs(w)
        if den == 0:
            return None
        return num / den

    def _update_state(self, state: str, fomo: Optional[float], eup: Optional[float], cap: Optional[float], hyst: float) -> str:
        # Acquire thresholds
        f_thr = float(self._thr.get('fomo_z', 1.5))
        e_thr = float(self._thr.get('euphoria_z', 2.0))
        c_thr = float(self._thr.get('capitulation_z', 2.0))
        # Entry conditions
        if fomo is not None and fomo >= f_thr:
            state = 'fomo'
        if eup is not None and eup >= e_thr and (state != 'fomo' or eup >= fomo):
            state = 'euphoria'
        if cap is not None and cap >= c_thr:
            state = 'capitulation'
        # Exit hysteresis
        if state == 'fomo' and (fomo is None or fomo < f_thr - hyst):
            state = ''
        if state == 'euphoria' and (eup is None or eup < e_thr - hyst):
            state = ''
        if state == 'capitulation' and (cap is None or cap < c_thr - hyst):
            state = ''
        return state

    def _policy_decision(self, fomo: Optional[float], eup: Optional[float], cap: Optional[float], weekend: int, holiday: int):
        veto_cfg = self._policy.get('veto', {})
        damp_cfg = self._policy.get('dampen', {})
        base_mult = float(self._policy.get('base_size_mult', 1.0))
        size_mult = base_mult
        flags = []
        reason = ''
        action = 'ENTER'
        def _hit(score: Optional[float], key: str, cfg: Dict[str, float]):
            thr = cfg.get(key)
            return score is not None and thr is not None and score >= float(thr)
        if _hit(eup, 'euphoria_z', veto_cfg):
            action = 'VETO'; reason = 'EUPHORIA'
        elif _hit(fomo, 'fomo_z', veto_cfg):
            action = 'VETO'; reason = 'FOMO_SPIKE'
        elif _hit(cap, 'capitulation_z', veto_cfg):
            action = 'VETO'; reason = 'CAPITULATION'
        # Dampen if not vetoed
        if action != 'VETO':
            damp_reason_parts = []
            if _hit(fomo, 'fomo_z', damp_cfg):
                damp_reason_parts.append('fomo')
            if _hit(eup, 'euphoria_z', damp_cfg):
                damp_reason_parts.append('euphoria')
            if _hit(cap, 'capitulation_z', damp_cfg):
                damp_reason_parts.append('capitulation')
            if damp_reason_parts:
                action = 'DAMPEN'
                size_mult *= float(damp_cfg.get('size_mult', 0.6))
                reason = 'DAMPEN:' + '+'.join(damp_reason_parts)
        # Weekend / holiday penalties multiplicative (applied even if dampen)
        if weekend and self._policy.get('weekend_penalty'):
            size_mult *= float(self._policy['weekend_penalty'])
            flags.append('weekend')
        if holiday and self._policy.get('holiday_penalty'):
            size_mult *= float(self._policy['holiday_penalty'])
            flags.append('holiday')
        if fomo and fomo > 0:
            flags.append(f'fomo{fomo:.1f}σ')
        if eup and eup > 0:
            flags.append(f'eup{eup:.1f}σ')
        if cap and cap > 0:
            flags.append(f'cap{cap:.1f}σ')
        return action, round(size_mult,3), reason or action, flags

    # -------------------- Holiday & Persistence -----------------------
    def _load_holidays(self, path: Path) -> set[str]:
        if not path.exists():
            return set()
        try:
            data = yaml.safe_load(path.read_text()) or {}
            dates = set()
            # Expect format: {region: [YYYY-MM-DD, ...], ...} or simple list
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        dates.update(v)
            elif isinstance(data, list):
                dates.update(data)
            return dates
        except Exception as e:
            logger.warning('Failed loading holidays: {}', e)
            return set()

    def _load_expectancy(self) -> None:
        try:
            if self._expectancy_path.exists():
                df = pd.read_parquet(self._expectancy_path)
                self._hour_tally = {int(r.hour): int(r.count) for r in df.itertuples() if r.kind == 'hour'}
                self._session_tally = {str(r.session): int(r.count) for r in df.itertuples() if r.kind == 'session'}
                self._total_observations = int(df['count'].sum())
        except Exception as e:
            logger.warning('Failed loading behavior expectancy: {}', e)

    def _save_expectancy(self) -> None:
        try:
            rows = []
            for h,c in self._hour_tally.items():
                rows.append({'kind':'hour','hour':h,'count':c,'session':None})
            for s,c in self._session_tally.items():
                rows.append({'kind':'session','hour':None,'count':c,'session':s})
            if not rows:
                return
            df = pd.DataFrame(rows)
            df.to_parquet(self._expectancy_path, index=False)
        except Exception as e:
            logger.warning('Failed saving behavior expectancy: {}', e)

    # -------------------- Divergence & Probability --------------------
    def _detect_divergence(self, ret: Optional[float], fm) -> int:
        try:
            if ret is None or fm is None:
                return 0
            cvd_chg = getattr(fm, 'cvd_chg', None)
            if cvd_chg is None:
                return 0
            # Divergence if price move sign != cvd_chg sign and |ret|>threshold
            if abs(ret) >= 0.002 and (ret > 0 > cvd_chg or ret < 0 < cvd_chg):
                return 1
            return 0
        except Exception:
            return 0

    def _estimate_prob(self, raw_score: float) -> float:
        try:
            if self._prob_model:
                # placeholder API: model.predict_proba([[raw_score]])[0][1]
                return float(self._prob_model.predict_proba([[raw_score]])[0][1])  # type: ignore
            # logistic fallback
            k = 1.2; x0 = 2.0
            import math as _m
            p = 1.0 / (1.0 + _m.exp(-k*(raw_score - x0))) if raw_score is not None else 0.0
            return float(round(p,4))
        except Exception:
            return 0.0

    # Calibration harness (external injection of outcomes)
    def record_outcome(self, raw_score: float, veto_triggered: int):
        try:
            self._calib_samples.append((raw_score, int(bool(veto_triggered))))
            if len(self._calib_samples) >= 200 and len(self._calib_samples) % 200 == 0:
                self._recalibrate()
        except Exception:
            pass

    def _recalibrate(self):  # simplistic logistic fit using pandas / numpy
        try:
            import numpy as np
            xs = np.array([s for s,_ in self._calib_samples]).reshape(-1,1)
            ys = np.array([y for _,y in self._calib_samples])
            from sklearn.linear_model import LogisticRegression
            mdl = LogisticRegression()
            mdl.fit(xs, ys)
            self._prob_model = mdl
            logger.info('BehaviorEngine calibrated on %d samples', len(xs))
        except Exception as e:
            logger.warning('Behavior calibration failed: {}', e)

__all__ = ['BehaviorEngine']
