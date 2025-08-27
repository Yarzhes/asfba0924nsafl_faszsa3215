"""Bracket / OCO construction.
Generates reduce-only bracket child orders (stop + TP ladder) plus BE / trailing rules metadata.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class BracketLeg:
    kind: str  # SL|TP
    price: float
    size: float
    reduce_only: bool = True

@dataclass
class Brackets:
    stop: BracketLeg
    tps: List[BracketLeg]
    meta: Dict[str, Any]


def build_brackets(entry_px: float, side: str, atr: float | None, size: float, execution_cfg: Dict[str, Any]) -> Optional[Brackets]:
    bcfg = ((execution_cfg or {}).get('brackets') or {})
    if not bcfg.get('enabled', True):
        return None
    stop_mult = float(bcfg.get('stop_atr_mult', 1.4))
    tp_mults = list(bcfg.get('tp_atr_mults', [1.8,2.6,3.5]))
    tp_scales = list(bcfg.get('tp_scales', [0.5,0.3,0.2]))
    # Normalize scales -> sum to 1.0
    ssum = sum(tp_scales) or 1.0
    tp_scales = [x/ssum for x in tp_scales]
    if atr is None or atr <= 0:
        return None
    up = side.upper()=='LONG'
    stop_px = entry_px - stop_mult*atr if up else entry_px + stop_mult*atr
    tps: List[BracketLeg] = []
    for mult, scale in zip(tp_mults, tp_scales):
        tp_px = entry_px + mult*atr if up else entry_px - mult*atr
        tps.append(BracketLeg('TP', tp_px, size*scale, True))
    stop_leg = BracketLeg('SL', stop_px, size, True)
    meta = {
        'be': (bcfg.get('break_even') or {}),
        'trailing': (bcfg.get('trailing') or {}),
        'config': {'stop_mult':stop_mult,'tp_mults':tp_mults,'tp_scales':tp_scales}
    }
    # Runtime state flags
    meta.setdefault('state', {'be_done': False, 'trail_armed': False})
    return Brackets(stop_leg, tps, meta)


def update_brackets(brackets: Brackets, *, current_price: float, entry_px: float, side: str, atr: float, tick_size: float) -> bool:
    """Apply BE + trailing logic in-place. Returns True if stop modified.

    Break-even: when move >= be_trigger_atr * ATR -> move stop to entry +/- be_lock_ticks * tick_size.
    Trailing  : after arm_atr * ATR move, trail stop by trail_atr_mult * ATR.
    """
    changed = False
    if atr is None or atr <= 0:
        return False
    up = side.upper() == 'LONG'
    state = brackets.meta.setdefault('state', {'be_done': False, 'trail_armed': False})
    be_cfg = (brackets.meta.get('be') or {}) if brackets.meta else {}
    tr_cfg = (brackets.meta.get('trailing') or {}) if brackets.meta else {}
    move = (current_price - entry_px) if up else (entry_px - current_price)
    # Break-even move
    if be_cfg.get('enabled', True) and not state.get('be_done'):
        be_trigger_atr = float(be_cfg.get('be_trigger_atr', 1.2))
        if move >= be_trigger_atr * atr:
            lock_ticks = int(be_cfg.get('be_lock_ticks', 0))
            new_stop = entry_px + lock_ticks * tick_size if up else entry_px - lock_ticks * tick_size
            if up and new_stop > brackets.stop.price:
                brackets.stop.price = new_stop; changed = True
            elif not up and new_stop < brackets.stop.price:
                brackets.stop.price = new_stop; changed = True
            state['be_done'] = True
    # Trailing arming
    if tr_cfg.get('enabled', True):
        arm_atr = float(tr_cfg.get('arm_atr', 2.0))
        if move >= arm_atr * atr:
            state['trail_armed'] = True
    # Apply trailing if armed
        if state.get('trail_armed'):
            trail_mult = float(tr_cfg.get('trail_atr_mult', 1.0))
            desired_stop = current_price - trail_mult * atr if up else current_price + trail_mult * atr
            if up and desired_stop > brackets.stop.price:
                brackets.stop.price = desired_stop; changed = True
            elif not up and desired_stop < brackets.stop.price:
                brackets.stop.price = desired_stop; changed = True
    return changed

__all__ = ['build_brackets','BracketLeg','Brackets','update_brackets']
