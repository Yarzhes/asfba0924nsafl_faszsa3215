"""Parameter search spaces for auto-optimization (bounded & safe).

We define small, meaningful ranges to reduce overfit risk. Provides
sampling helpers compatible with Optuna-like trial objects or a simple
RandomTrial shim for unit tests.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable
import random

@dataclass
class ParamDef:
    kind: str  # 'int','float','categorical','composite'
    name: str
    bounds: Tuple[Any, Any] | None = None
    choices: List[Any] | None = None
    step: int | None = None
    sampler: Callable[[Any], Any] | None = None  # custom

class AutoOptSpace:
    def __init__(self):
        self._params: List[ParamDef] = []
        self._build_default()

    def _add(self, *args, **kwargs):
        self._params.append(ParamDef(*args, **kwargs))

    def _build_default(self):
        # Ensemble
        self._add('float','ensemble.vote_threshold_trend', (0.2,0.8))
        self._add('float','ensemble.vote_threshold_mean_revert', (0.2,0.8))
        self._add('categorical','ensemble.min_agree_trend', choices=[1,2,3])
        self._add('categorical','ensemble.min_agree_mean_revert', choices=[1,2,3])
        self._add('float','ensemble.margin_of_victory', (0.05,0.35))
        self._add('float','ensemble.confidence_floor', (0.4,0.8))
        # Execution
        self._add('categorical','execution.k1_ticks', choices=[0,1,2])
        self._add('float','execution.taker_fallback_ms', (600,3000))
        self._add('float','execution.max_chase_bps', (4,15))
        self._add('float','execution.stop_atr_mult', (1.0,2.2))
        # Composite triple (monotonic) captured via custom sampler producing three values
        def _tp_sampler(trial):
            m1 = self._sample_float(trial,'execution.tp1_atr_mult',(1.4,2.2))
            m2_low = max(m1+0.2,2.0)
            m2 = self._sample_float(trial,'execution.tp2_atr_mult',(m2_low,3.0))
            m3_low = max(m2+0.2,2.6)
            m3 = self._sample_float(trial,'execution.tp3_atr_mult',(m3_low,4.0))
            return (m1,m2,m3)
        self._add('composite','execution.tp_atr_mults', sampler=_tp_sampler)
        # Hedging
        self._add('float','hedge.beta_band_min', (-0.25,-0.05))
        self._add('float','hedge.beta_band_max', (0.05,0.25))
        self._add('float','hedge.rebalance_min_notional', (0.002,0.02))
        self._add('float','hedge.corr_threshold_high', (0.4,0.7))
        # Risk filters
        self._add('float','risk.daily_loss_cap_pct', (1.5,4.0))
        self._add('categorical','risk.max_positions_total', choices=[2,3,4,5])
        # Sprint 30: MTC gate tuning (keep narrow to reduce variance)
        self._add('float','mtc.thresholds.confirm_full', (0.6,0.85))
        self._add('float','mtc.thresholds.confirm_partial', (0.4,0.7))
        self._add('float','mtc.rules.trend.adx_min', (15,28))
        self._add('float','mtc.rules.momentum.macd_slope_min', (0.0,0.6))
        self._add('float','mtc.rules.volatility.atr_pctile_max', (0.90,0.99))
        # RSI band edges (long lower & upper, short lower & upper) - sampled discretely to maintain coherence
        self._add('categorical','mtc.rules.momentum.rsi_band_long_lower', choices=[40,42,45])
        self._add('categorical','mtc.rules.momentum.rsi_band_long_upper', choices=[65,70,72])
        self._add('categorical','mtc.rules.momentum.rsi_band_short_lower', choices=[28,30,32])
        self._add('categorical','mtc.rules.momentum.rsi_band_short_upper', choices=[52,55,58])

    def sample(self, trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for p in self._params:
            if p.kind == 'float':
                params[p.name] = self._sample_float(trial,p.name,p.bounds)
            elif p.kind == 'int':
                low,high = p.bounds
                params[p.name] = getattr(trial,'suggest_int',self._rand_int)(p.name, int(low), int(high), p.step or 1)
            elif p.kind == 'categorical':
                params[p.name] = getattr(trial,'suggest_categorical',self._rand_choice)(p.name, p.choices)
            elif p.kind == 'composite':
                params[p.name] = p.sampler(trial)
        return params

    # --- helpers -----------------------------------------------------------
    def _sample_float(self, trial, name, bounds):
        low,high = bounds
        suggest = getattr(trial,'suggest_float',None)
        if suggest:
            return suggest(name,float(low),float(high))
        return random.uniform(low,high)
    def _rand_int(self,name,low,high,step):
        return random.randrange(low,high+1,step)
    def _rand_choice(self,name,choices):
        return random.choice(choices)

__all__ = ['AutoOptSpace']
