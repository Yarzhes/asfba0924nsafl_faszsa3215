"""Adaptive Exits Module (Sprint 34)

Generates dynamic stop-loss, target, trailing and partial take-profit
configuration per trade using ATR, regime context and recent swing
structure for confluence.

Design goals:
- Non-invasive: if disabled or inputs missing, returns None.
- Pure function style: stateless given inputs; caller stores state.
- Extensible: trailing + breakeven + partial handling metadata only;
  execution engine/backtester applies mechanics during bar loop.

Returned dict schema:
{
  'stop_price': float,
  'target_price': float,
  'partial_tp': [ {'price': float,'pct': float,'rr': float} ... ],
  'trail_config': {
       'type': 'atr'|'structure',
       'step': float,     # ATR multiple step for atr trailing
       'activated': False
   },
  'breakeven': {'enabled': bool,'trigger_rr': float},
  'meta': { 'atr': float,'atr_mult_stop_eff': float,'atr_mult_target_eff': float,'regime_profile': str,'struct_used': bool,'swing_stop': float,'rr_initial': float },
  'reason': 'atr_regime_struct_confluence'
}

Implementation notes:
- Swing detection: simple local highs/lows using last N bars highs/lows; if
  closest structural stop (for LONG => last swing low, for SHORT => last swing high)
  is closer (tighter) than ATR derived stop distance, we snap to it.
- Regime multipliers: if provided regime dict contains keys (profile or vol state)
  we adjust ATR multiples multiplicatively.
- Clamp: enforce min/max stop distance in % of current price.
- Partial TPs: convert rr levels (R multiples) to absolute price.

Limitations:
- Does not persist trailing progress; caller must update stop in runtime.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List

import math


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _calc_atr(ohlcv_tail, period: int) -> Optional[float]:
    try:
        if ohlcv_tail is None or len(ohlcv_tail) < max(2, period):
            return None
        # Expect DataFrame with columns high, low, close
        import pandas as pd
        df = ohlcv_tail.iloc[-period-1:].copy()
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        if atr is None or math.isnan(atr):
            return None
        return float(atr)
    except Exception:
        return None


def _find_swings(ohlcv_tail, lookback: int) -> Dict[str, Optional[float]]:
    swing_low = swing_high = None
    try:
        if ohlcv_tail is None or len(ohlcv_tail) < lookback:
            return {'swing_low': None, 'swing_high': None}
        window = ohlcv_tail.iloc[-lookback:]
        swing_low = float(window['low'].min())
        swing_high = float(window['high'].max())
    except Exception:
        pass
    return {'swing_low': swing_low, 'swing_high': swing_high}


def generate_adaptive_exits(
    symbol: str,
    side: str,
    price: float,
    ohlcv_tail,              # pandas DataFrame recent bars including current
    regime_info: Dict[str, Any],  # expected keys: profile, vol_state
    settings: Dict[str, Any],
    atr_current: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    cfg = (((settings or {}).get('risk') or {}).get('adaptive_exits') or {})
    if not cfg.get('enabled', False):
        return None

    side_up = str(side).upper()
    if side_up not in ('LONG','SHORT'):
        return None

    price = _safe_float(price, 0.0)
    if price <= 0:
        return None

    atr_lb = int(cfg.get('atr_lookback', 14))
    atr = atr_current if atr_current and atr_current > 0 else _calc_atr(ohlcv_tail, atr_lb)
    if atr is None or atr <= 0:
        # Fallback small synthetic ATR (0.4% of price)
        atr = price * 0.004

    atr_mult_stop = float(cfg.get('atr_mult_stop', 1.2))
    atr_mult_target = float(cfg.get('atr_mult_target', 2.5))

    # Optional Sprint 37 optimized stop override (distance only)
    try:
        if (settings.get('auto_stop_opt') or {}).get('enabled'):
            from ultra_signals.opt.stop_resolver import resolve_stop  # lazy import
            dist = resolve_stop(symbol, (regime_info or {}).get('tf','5m'), profile, atr, price, settings)  # regime profile as key
            if dist and dist > 0:
                # convert distance back to implied atr multiple for meta (if atr>0)
                stop_distance = dist
                if atr > 0:
                    atr_mult_stop = stop_distance / atr
    except Exception:
        stop_distance = None  # will be recalculated below if fails

    # Regime adjustments
    reg_mults = cfg.get('regime_multiplier', {}) or {}
    profile = (regime_info or {}).get('profile') or (regime_info or {}).get('primary')
    vol_state = (regime_info or {}).get('vol_state') or (regime_info or {}).get('vol')

    def _apply_reg(mult):
        m_eff = 1.0
        # trending / chop semantics
        if profile and str(profile).startswith('trend'):
            m_eff *= float(reg_mults.get('trending', 1.0))
        if profile and 'chop' in str(profile):
            m_eff *= float(reg_mults.get('chop', 1.0))
        if vol_state and 'high' in str(vol_state):
            m_eff *= float(reg_mults.get('high_vol', 1.0))
        if vol_state and ('low' in str(vol_state) or 'crush' in str(vol_state)):
            m_eff *= float(reg_mults.get('low_vol', 1.0))
        return mult * m_eff

    atr_mult_stop_eff = _apply_reg(atr_mult_stop)
    atr_mult_target_eff = _apply_reg(atr_mult_target)

    # If optimized stop distance not already derived, compute from ATR multiple
    if 'stop_distance' not in locals() or stop_distance is None:
        stop_distance = atr * atr_mult_stop_eff
    target_distance = atr * atr_mult_target_eff

    # Clamp stop distance
    min_stop_pct = float(cfg.get('min_stop_pct', 0.003))
    max_stop_pct = float(cfg.get('max_stop_pct', 0.02))
    stop_distance = max(min_stop_pct * price, min(stop_distance, max_stop_pct * price))

    # Structural snapping
    struct_used = False
    swing_stop = None
    if cfg.get('structural_confluence', True):
        sw = _find_swings(ohlcv_tail, int(cfg.get('swing_lookback', 12)))
        if side_up == 'LONG' and sw.get('swing_low'):
            candidate = float(sw['swing_low'])
            dist = price - candidate
            if dist > 0 and dist < stop_distance:  # closer (tighter)
                stop_distance = dist
                swing_stop = candidate
                struct_used = True
        elif side_up == 'SHORT' and sw.get('swing_high'):
            candidate = float(sw['swing_high'])
            dist = candidate - price
            if dist > 0 and dist < stop_distance:
                stop_distance = dist
                swing_stop = candidate
                struct_used = True

    if side_up == 'LONG':
        stop_price = price - stop_distance
        target_price = price + target_distance
    else:
        stop_price = price + stop_distance
        target_price = price - target_distance

    # Initial RR (approx)
    rr_initial = abs(target_price - price) / max(1e-9, abs(price - stop_price))

    # Breakeven config
    breakeven = {
        'enabled': bool(cfg.get('breakeven_enable', True)),
        'trigger_rr': float(cfg.get('breakeven_trigger_rr', 1.2))
    }

    # Trailing config
    trail_cfg = {
        'enabled': bool(cfg.get('trailing_enable', True)),
        'type': str(cfg.get('trailing_type','atr')),
        'step': float(cfg.get('trailing_step_mult', 1.0)) * atr,
        'activated': False
    }

    # Partial TPs
    partials: List[Dict[str, Any]] = []
    pt_cfg = cfg.get('partial_tp') or {}
    if pt_cfg.get('enabled') and isinstance(pt_cfg.get('levels'), list):
        for lvl in pt_cfg['levels']:
            try:
                rr = float(lvl.get('rr'))
                pct = float(lvl.get('pct'))
                if rr <= 0 or pct <= 0:
                    continue
                if side_up == 'LONG':
                    price_lvl = price + rr * (price - stop_price)
                else:
                    price_lvl = price - rr * (stop_price - price)
                partials.append({'price': price_lvl, 'pct': pct, 'rr': rr})
            except Exception:
                continue

    return {
        'stop_price': float(stop_price),
        'target_price': float(target_price),
        'partial_tp': partials,
        'trail_config': trail_cfg,
        'breakeven': breakeven,
        'meta': {
            'atr': atr,
            'atr_mult_stop_eff': atr_mult_stop_eff,
            'atr_mult_target_eff': atr_mult_target_eff,
            'regime_profile': profile,
            'struct_used': struct_used,
            'swing_stop': swing_stop,
            'rr_initial': rr_initial,
        },
        'reason': 'atr_regime_struct_confluence'
    }

__all__ = ['generate_adaptive_exits']
