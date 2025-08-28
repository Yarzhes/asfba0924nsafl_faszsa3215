from __future__ import annotations
from typing import Optional, Dict, Any
import math
from loguru import logger
from .playbooks import (
    Playbook,
    make_trend_breakout, make_trend_pullback,
    make_mr_bollinger_fade, make_mr_vwap_revert, make_chop_flat
)


def _get_setting(s, key, default=None):
    """Utility: read setting key from either dict or Pydantic model."""
    if isinstance(s, dict):
        return s.get(key, default)
    try:
        return getattr(s, key, default)
    except Exception:
        return default

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
    # support both dict-style settings and Pydantic models
    def _get_setting(s, key, default=None):
        if isinstance(s, dict):
            return s.get(key, default)
        try:
            return getattr(s, key, default)
        except Exception:
            return default

    pb_cfg = (_get_setting(settings, 'playbooks') or {})
    # Prefer smoothed regime profile when available; fall back to profile attr
    regime_name = 'trend'
    try:
        if regime_obj is not None:
            # if smoothed probs exist, pick argmax
            sm = getattr(regime_obj, 'smoothed_regime_probs', None)
            if isinstance(sm, dict) and sm:
                regime_name = max(sm.items(), key=lambda kv: kv[1])[0]
            else:
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
    # expose playbook hint and confidence flag into plan metadata if present
    try:
        if regime_obj is not None:
            hint = getattr(regime_obj, 'playbook_hint', None)
            cf = getattr(regime_obj, 'regime_confidence_flag', None)
            if hint:
                # annotate plan with suggested playbook id for telemetry
                pass
    except Exception:
        pass
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

    # ATR for stop sizing & price level computation
    close_price = (features.get('ohlcv') or {}).get('close') or None
    atr = getattr(volatility, 'atr', None) if volatility else None
    if (atr is None or atr == 0) and close_price:
        # fallback: treat 1% of price as atr approx
        atr = close_price * 0.01
    if atr is None or atr == 0:
        # still nothing -> cannot build price-based plan, but return multipliers
        atr = None

    first_tp_mult = playbook.exit.tp_atr_mults[0] if playbook.exit.tp_atr_mults else (playbook.exit.stop_atr_mult * 1.4)
    exp_rr = _expected_rr(playbook.exit.stop_atr_mult, first_tp_mult)
    if playbook.risk.rr_min and exp_rr and exp_rr < playbook.risk.rr_min:
        return None

    # Compute concrete stop/tp price bands if we have price + atr
    stop_price = None
    tp_bands = []
    side = getattr(decision, 'decision', 'LONG')
    if close_price and atr:
        if side == 'LONG':
            stop_price = close_price - playbook.exit.stop_atr_mult * atr
            tp_bands = [close_price + m * atr for m in playbook.exit.tp_atr_mults]
        elif side == 'SHORT':
            stop_price = close_price + playbook.exit.stop_atr_mult * atr
            tp_bands = [close_price - m * atr for m in playbook.exit.tp_atr_mults]

    # Build plan (retain existing keys for backward compatibility; add new ones)
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
        'stop_price': stop_price,
        'tp_bands': tp_bands,
        'atr_used': atr,
    'playbook_hint': getattr(regime_obj, 'playbook_hint', None) if regime_obj is not None else None,
    'regime_confidence_flag': getattr(regime_obj, 'regime_confidence_flag', None) if regime_obj is not None else None,
    }

    # If execution settings prefer VWAP or the playbook suggests VWAP, annotate plan
    try:
        exec_cfg = _get_setting(settings, 'execution') or {}
        default_algo = exec_cfg.get('default_child_algo')
        if default_algo and str(default_algo).upper() == 'VWAP':
            plan['child_algo'] = 'VWAP'
            plan['vwap_cfg'] = exec_cfg.get('vwap', {})
        elif 'vwap' in (playbook.name or '').lower():
            plan['child_algo'] = 'VWAP'
            plan['vwap_cfg'] = (exec_cfg.get('vwap') or {})
    except Exception:
        pass

    # Enriched debug line (align closer to spec wording)
    adx_val = getattr(trend, 'adx', None) if trend else None
    ema_sep_val = _ema_separation_atr(features)
    try:
        logger.debug(
            '[PLAN] pb={} side={} conf={:.2f} adx={} ema_sep_atr={} rr={} size_scale={} trail@{} timeout={} stop_mult={} tps={} atr={}',
            playbook.name,
            side,
            decision.confidence,
            round(adx_val, 2) if adx_val is not None else None,
            round(ema_sep_val, 3) if ema_sep_val is not None else None,
            round(exp_rr, 3) if exp_rr else None,
            playbook.risk.size_scale,
            playbook.exit.trail_after_rr,
            playbook.exit.timeout_bars,
            playbook.exit.stop_atr_mult,
            playbook.exit.tp_atr_mults,
            round(atr, 4) if atr else None,
        )
    except Exception:
        pass
    return plan


def is_plan_timed_out(plan: Dict[str, Any], bars_since_creation: int, realized_rr: float | None) -> bool:
    """Utility for backtest/runtime loops to decide if an open (unfilled or stale)
    execution plan should be cancelled due to timeout.

    Logic (simple, spec-aligned):
      - If plan has timeout_bars and bars_since_creation >= timeout_bars
        AND realized_rr (progress toward first TP) < 0.1 RR -> timed out.
      - If no timeout_bars configured, never times out here.
    """
    if not plan or not isinstance(plan, dict):
        return False
    tb = plan.get('timeout_bars')
    if not tb or tb <= 0:
        return False
    try:
        if bars_since_creation >= int(tb):
            prog = float(realized_rr or 0.0)
            return prog < 0.10
    except Exception:
        return False
    return False
