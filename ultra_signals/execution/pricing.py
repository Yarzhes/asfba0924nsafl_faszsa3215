"""Smart limit pricing & fences (maker-first with taker fallback).

Primary entrypoint: build_exec_plan(...)
Returns ExecPlan describing initial order plus taker fallback policy.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import math

@dataclass
class ExecPlan:
    symbol: str
    side: Literal['LONG','SHORT']
    price: float
    post_only: bool
    ts_ms: int
    taker_fallback_after_ms: int
    taker_price: Optional[float]
    fence_reason: Optional[str] = None
    meta: Dict[str, Any] | None = None

    def to_order(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'price': self.price,
            'type': 'LIMIT',
            'post_only': self.post_only,
            'ts': self.ts_ms,
            'taker_fallback_after_ms': self.taker_fallback_after_ms,
            'taker_price': self.taker_price,
            'fence_reason': self.fence_reason,
        }

TICKS_PER_PCT = 10000  # fallback scaling if tick not provided

def _apply_anchor(anchor: str, best_bid: float, best_ask: float, mid: float, vwap: Optional[float], close: Optional[float]) -> float:
    if anchor == 'mid':
        return mid
    if anchor == 'vwap' and vwap:
        return float(vwap)
    if anchor == 'close' and close:
        return float(close)
    return mid

def build_exec_plan(symbol: str, side: str, book: Dict[str,float], *,
                     tick_size: float,
                     atr: Optional[float],
                     atr_pct: Optional[float],
                     regime: Optional[str],
                     settings: Dict[str, Any],
                     now_ms: int,
                     vwap: Optional[float]=None,
                     last_close: Optional[float]=None,
                     cvd_flip: bool=False) -> ExecPlan | None:
    ex_cfg = (settings.get('execution') or {}) if isinstance(settings, dict) else getattr(settings,'execution',{}) or {}
    maker_first = ex_cfg.get('maker_first', True)
    k1_ticks = int(ex_cfg.get('k1_ticks',1))
    k2_ticks = int(ex_cfg.get('taker_offset_ticks',1))
    fallback_ms = int(ex_cfg.get('taker_fallback_ms',1200))
    max_spread_pct = float(ex_cfg.get('max_spread_pct',0.06))
    max_chase_bps = float(ex_cfg.get('max_chase_bps',8))
    atr_pct_limit = float(ex_cfg.get('atr_pct_limit',0.97))
    price_anchor = ex_cfg.get('price_anchor','mid')
    # Basic book fields
    bid = float(book['bid']); ask = float(book['ask'])
    spread = max(ask - bid, 0.0)
    mid = (ask + bid)/2.0 if ask and bid else (ask or bid)
    if not mid:
        return None
    spread_pct = spread / mid * 100.0 if mid else 0.0
    if spread_pct > max_spread_pct*100.0:
        return ExecPlan(symbol, side.upper(), mid, False, now_ms, 0, None, fence_reason='spread', meta={'spread_pct':spread_pct})
    if atr_pct is not None and atr_pct > atr_pct_limit:
        return ExecPlan(symbol, side.upper(), mid, False, now_ms, 0, None, fence_reason='atr_pct', meta={'atr_pct':atr_pct})
    if cvd_flip:
        return ExecPlan(symbol, side.upper(), mid, False, now_ms, 0, None, fence_reason='cvd_flip')
    anchor_px = _apply_anchor(price_anchor, bid, ask, mid, vwap, last_close)
    # Price fence vs anchor (do not chase more than max_chase_bps)
    def _bps(a,b):
        try:
            return abs(a-b)/b*10_000.0
        except Exception:
            return 0.0
    # Initial maker price
    if side.upper()=='LONG':
        limit_px = bid + k1_ticks * tick_size if maker_first else ask + k2_ticks * tick_size
        taker_px = ask + k2_ticks * tick_size
    else:
        limit_px = ask - k1_ticks * tick_size if maker_first else bid - k2_ticks * tick_size
        taker_px = bid - k2_ticks * tick_size
    chase_bps = _bps(limit_px, anchor_px)
    if chase_bps > max_chase_bps:
        return ExecPlan(symbol, side.upper(), limit_px, False, now_ms, 0, None, fence_reason='chase', meta={'chase_bps':chase_bps})
    plan = ExecPlan(symbol, side.upper(), limit_px, maker_first, now_ms, fallback_ms if maker_first else 0, taker_px if maker_first else None)
    return plan

__all__ = ['ExecPlan','build_exec_plan']
