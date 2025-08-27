"""Child order execution algos (TWAP, Iceberg, POV prototype).

Functions return a list of ExecPlan-like dicts; actual pricing delegated to pricing.build_exec_plan.
Simplified: uses notional threshold & provided size to split.
"""
from __future__ import annotations
from typing import List, Dict, Any
import math
from .pricing import ExecPlan, build_exec_plan


def plan_child_orders(symbol: str, side: str, total_notional: float, price: float, *, settings: Dict[str, Any], book: Dict[str,float], tick_size: float, atr: float | None, atr_pct: float | None, regime: str | None, now_ms: int) -> List[ExecPlan]:
    ex_cfg = settings.get('execution', {}) if isinstance(settings, dict) else {}
    algos = ex_cfg.get('algos', {})
    threshold = float(algos.get('threshold_usd', 20_000))
    if total_notional < threshold:
        # single plan
        plan = build_exec_plan(symbol, side, book, tick_size=tick_size, atr=atr, atr_pct=atr_pct, regime=regime, settings=settings, now_ms=now_ms)
        return [plan] if plan else []
    # choose algo heuristically
    regime_mode = (regime or 'trend').lower()
    use_iceberg = regime_mode == 'trend'
    plans: List[ExecPlan] = []
    if use_iceberg:
        icfg = algos.get('iceberg', {})
        clip = float(icfg.get('clip_usd', 3000.0))
        refresh_ms = int(icfg.get('refresh_ms',800))
        remaining = total_notional
        ix = 0
        while remaining > 0 and ix < 100:
            notional = min(clip, remaining)
            plan = build_exec_plan(symbol, side, book, tick_size=tick_size, atr=atr, atr_pct=atr_pct, regime=regime, settings=settings, now_ms=now_ms + ix*refresh_ms)
            if plan:
                plan.meta = (plan.meta or {})
                plan.meta.update({'child_ix':ix,'algo':'ICEBERG','child_notional':notional})
                plans.append(plan)
            remaining -= notional
            ix += 1
    else:  # TWAP
        tcfg = algos.get('twap', {})
        dur = int(tcfg.get('duration_s',120))*1000
        slices = int(tcfg.get('slices',6)) or 1
        per = dur//slices
        for i in range(slices):
            plan = build_exec_plan(symbol, side, book, tick_size=tick_size, atr=atr, atr_pct=atr_pct, regime=regime, settings=settings, now_ms=now_ms + i*per)
            if plan:
                plan.meta = (plan.meta or {})
                plan.meta.update({'child_ix':i,'algo':'TWAP','slices':slices})
                plans.append(plan)
    return plans

__all__ = ['plan_child_orders']
