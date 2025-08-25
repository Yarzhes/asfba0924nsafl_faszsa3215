from __future__ import annotations
from typing import Optional, Dict, Any
import math
from loguru import logger
from .playbooks import (
    Playbook, FACTORY_MAP,
    make_trend_breakout, make_trend_pullback,
    make_mr_bollinger_fade, make_mr_vwap_revert, make_chop_flat
)

# Simple heuristic selectors ---------------------------------------------------

def _ema_separation_atr(features: Dict[str, Any]) -> Optional[float]:
    trend = features.get('trend')
    vol = features.get('volatility')
    if not trend or not vol:
        return None
    try:
        ema_s = getattr(trend, 'ema_short', None)
        ema_l = getattr(trend, 'ema_long', None)
        atr = getattr(vol, 'atr', None)
        if None in (ema_s, ema_l, atr) or atr == 0:
            return None
        return abs((ema_s - ema_l) / atr)
    except Exception:
        return None

def _expected_rr(stop_mult: float, first_tp_mult: float) -> Optional[float]:
    if stop_mult <= 0:
        return None
    return first_tp_mult / stop_mult


def select_playbook(regime_obj, features: Dict[str, Any], decision, settings: Dict) -> Optional[Playbook]:
    pb_cfg = (settings.get('playbooks') or {})
    regime_name = 'trend'
    try:
        if regime_obj is not None:
            regime_name = getattr(regime_obj, 'profile', None) or getattr(regime_obj, 'mode', None) or 'trend'
            if hasattr(regime_name, 'value'):
                regime_name = regime_name.value
    except Exception:
        pass

    # Chop -> flat directly
    if regime_name == 'chop':
        flat_cfg = (((pb_cfg.get('chop') or {}).get('flat')) or {})
        return make_chop_flat(flat_cfg)

    # Trend logic: use squeeze / ema separation heuristic to pick breakout vs pullback
    # For simplicity: if ema_sep_atr >= threshold of breakout config -> breakout else pullback.
    ema_sep = _ema_separation_atr(features)
    trend_cfg = pb_cfg.get('trend') or {}
    if regime_name == 'trend' and trend_cfg:
        breakout_cfg = trend_cfg.get('breakout') or {}
        pullback_cfg = trend_cfg.get('pullback') or {}
        breakout_thr = (breakout_cfg.get('entry') or {}).get('ema_sep_atr_min', 0.30)
        if ema_sep is not None and ema_sep >= breakout_thr:
            if breakout_cfg.get('enabled', True):
                return make_trend_breakout(breakout_cfg)
        if pullback_cfg.get('enabled', True):
            return make_trend_pullback(pullback_cfg)

    # Mean reversion: choose bb_fade vs vwap_revert based on presence of rsi extreme
    mr_cfg = pb_cfg.get('mean_revert') or {}
    if regime_name == 'mean_revert' and mr_cfg:
        bb_cfg = mr_cfg.get('bb_fade') or {}
        vwap_cfg = mr_cfg.get('vwap_revert') or {}
        rsi = getattr(getattr(features.get('momentum'), 'rsi', None), 'rsi', None) if False else getattr(features.get('momentum'), 'rsi', None)
        if rsi is not None and (bb_cfg.get('entry') or {}).get('rsi_extreme') and bb_cfg.get('enabled', True):
            low, high = (bb_cfg['entry']['rsi_extreme'])
            if rsi <= low or rsi >= high:
                return make_mr_bollinger_fade(bb_cfg)
        if vwap_cfg.get('enabled', True):
            return make_mr_vwap_revert(vwap_cfg)

    return None


def _orderflow_pass(of_confirm: dict, of_detail: dict, decision_side: str) -> bool:
    if not of_confirm:
        return True
    if not of_detail:
        return False
    # cvd threshold
    if 'cvd' in of_confirm:
        cvd_req = float(of_confirm['cvd'])
        cvd_val = of_detail.get('cvd_chg') or of_detail.get('cvd')
        if cvd_val is None:
            return False
        if decision_side == 'LONG' and cvd_val < cvd_req:
            return False
        if decision_side == 'SHORT' and cvd_val > -cvd_req:
            return False
    # sweep presence
    if of_confirm.get('sweep') and not of_detail.get('sweep_flag'):
        return False
    # liquidation cluster side (simple containment)
    if 'liq_cluster_side' in of_confirm:
        allowed = str(of_confirm['liq_cluster_side']).split('|')
        dom = of_detail.get('liq_dom')
        if dom is None:
            return False
        token = f"{dom}_liq"
        if token not in allowed:
            return False
    return True


def build_plan(playbook: Playbook, features: Dict[str, Any], decision, settings: Dict) -> Optional[Dict[str, Any]]:
    if not playbook or playbook.abstain:
        return None
    # Gates
    if decision.confidence < playbook.entry.min_conf:
        return None
    trend = features.get('trend')
    volatility = features.get('volatility')
    momentum = features.get('momentum')
    regime_obj = features.get('regime')
    of_detail = (decision.vote_detail or {}).get('orderflow') if decision.vote_detail else {}

    # ADX gate
    if playbook.entry.min_adx is not None:
        adx = getattr(trend, 'adx', None) if trend else None
        if adx is None or adx < playbook.entry.min_adx:
            return None
    # EMA separation gate
    if playbook.entry.ema_sep_atr_min is not None:
        ema_sep = _ema_separation_atr(features)
        if ema_sep is None or ema_sep < playbook.entry.ema_sep_atr_min:
            return None
    # RSI extremes
    if playbook.entry.rsi_extreme and momentum:
        rsi = getattr(momentum, 'rsi', None)
        low, high = playbook.entry.rsi_extreme
        if rsi is None or (rsi > low and rsi < high):
            return None
    # Orderflow confirmations
    if not _orderflow_pass(playbook.entry.of_confirm or {}, of_detail or {}, decision.decision):
        return None

    # ATR for stop sizing
    atr = getattr(volatility, 'atr', None) if volatility else None
    if atr is None or atr == 0:
        # fallback: treat 1% of price as atr approx
        close = (features.get('ohlcv') or {}).get('close') or 1.0
        atr = close * 0.01
    stop_dist = playbook.exit.stop_atr_mult * atr
    first_tp_mult = playbook.exit.tp_atr_mults[0] if playbook.exit.tp_atr_mults else (playbook.exit.stop_atr_mult * 1.4)
    exp_rr = _expected_rr(playbook.exit.stop_atr_mult, first_tp_mult)
    if playbook.risk.rr_min and exp_rr and exp_rr < playbook.risk.rr_min:
        return None

    plan = {
        'entry_type': 'market',
        'stop_atr_mult': playbook.exit.stop_atr_mult,
        'tp_atr_mults': playbook.exit.tp_atr_mults,
        'trail_after_rr': playbook.exit.trail_after_rr,
        'timeout_bars': playbook.exit.timeout_bars,
        'size_scale': playbook.risk.size_scale,
        'cooldown_bars': playbook.risk.cooldown_bars,
        'reason': playbook.name,
        'expected_rr': exp_rr,
    }
    logger.debug('[PLAN] pb={} conf={:.2f} exp_rr={} size_scale={} trail@{} timeout={} tp={} stop_mult={}',
                 playbook.name, decision.confidence, round(exp_rr,3) if exp_rr else None, playbook.risk.size_scale,
                 playbook.exit.trail_after_rr, playbook.exit.timeout_bars, playbook.exit.tp_atr_mults, playbook.exit.stop_atr_mult)
    return plan
