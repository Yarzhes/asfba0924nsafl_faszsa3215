"""Whale / Smart Money Veto & Boost Gate (Sprint 41)

Evaluates latest WhaleFeatures snapshot and maps extreme conditions to
VETO / DAMPEN / BOOST actions for sizing & decision filters.

Inputs:
  - whale_features (WhaleFeatures instance or dict)
  - settings['features']['whale_risk'] config

Decision precedence:
  1. Explicit extreme veto (block sell extreme or composite pressure below veto thr)
  2. Deposit spike action
  3. Withdrawal spike action / composite boost
  4. Block buy extreme action
  5. Default NONE

BOOST returns size_mult>1; DAMPEN <1.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(slots=True)
class WhaleGateDecision:
    action: str                  # VETO | DAMPEN | BOOST | NONE
    reason: Optional[str] = None
    size_mult: Optional[float] = None
    meta: Dict[str, Any] = None


def evaluate_whale_gate(whale_features: Any, settings: Dict[str, Any]) -> WhaleGateDecision:
    cfg = (((settings or {}).get('features') or {}).get('whale_risk') or {})
    if not cfg.get('enabled', True):
        return WhaleGateDecision(action='NONE', meta={'disabled': True})
    wf = whale_features
    if wf is None:
        return WhaleGateDecision(action='NONE', reason='NO_WHALE_DATA')
    if hasattr(wf, 'model_dump'):
        try:
            wf = wf.model_dump(exclude_none=True)
        except Exception:
            wf = dict(wf.__dict__)
    # Helper
    def _flag(name):
        v = wf.get(name)
        return bool(v) and int(v) == 1

    comp = wf.get('composite_pressure_score')
    block_z = wf.get('block_trade_notional_p99_z') or 0.0
    deposit_spike = _flag('exch_deposit_burst_flag')
    withdrawal_spike = _flag('exch_withdrawal_burst_flag')
    sweep_sell = _flag('sweep_sell_flag')
    sweep_buy = _flag('sweep_buy_flag')

    veto_thr = float(cfg.get('composite_pressure_veto_thr', -5_000_000))
    boost_thr = float(cfg.get('composite_pressure_boost_thr', 5_000_000))
    boost_size = float(cfg.get('boost_size_mult', 1.25))
    damp_size = float(cfg.get('dampen_size_mult', 0.7))

    # 1. Block sell extreme or composite deep negative veto
    if block_z is not None and block_z < -2.5 and cfg.get('block_sell_extreme_action') == 'VETO':
        return WhaleGateDecision(action='VETO', reason='BLOCK_SELL_EXTREME', meta={'block_z': block_z})
    if comp is not None and comp < veto_thr:
        return WhaleGateDecision(action='VETO', reason='NEG_SM_PRESSURE', meta={'comp': comp})

    # 2. Deposit spike (bearish) -> action mapping
    if deposit_spike:
        act = cfg.get('deposit_spike_action','DAMPEN')
        if act == 'VETO':
            return WhaleGateDecision(action='VETO', reason='DEPOSIT_SPIKE')
        if act == 'DAMPEN':
            return WhaleGateDecision(action='DAMPEN', reason='DEPOSIT_SPIKE', size_mult=damp_size)

    # 3. Withdrawal spike (bullish)
    if withdrawal_spike:
        act = cfg.get('withdrawal_spike_action','BOOST')
        if act == 'BOOST':
            return WhaleGateDecision(action='BOOST', reason='WITHDRAWAL_SPIKE', size_mult=boost_size)
        if act == 'DAMPEN':
            return WhaleGateDecision(action='DAMPEN', reason='WITHDRAWAL_SPIKE', size_mult=damp_size)
        if act == 'VETO':
            return WhaleGateDecision(action='VETO', reason='WITHDRAWAL_SPIKE')

    # 4. Composite boost or block buy extreme
    if comp is not None and comp > boost_thr:
        return WhaleGateDecision(action='BOOST', reason='POS_SM_PRESSURE', size_mult=boost_size)
    if block_z is not None and block_z > 2.5 and cfg.get('block_buy_extreme_action') == 'BOOST':
        return WhaleGateDecision(action='BOOST', reason='BLOCK_BUY_EXTREME', size_mult=boost_size)

    return WhaleGateDecision(action='NONE')

__all__ = ['WhaleGateDecision','evaluate_whale_gate']