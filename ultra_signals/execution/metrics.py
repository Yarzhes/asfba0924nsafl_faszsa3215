"""Execution attribution metrics helpers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ExecFill:
    px: float
    qty: float
    maker: bool
    ts_ms: int

@dataclass
class ExecAttribution:
    paper_entry_px: float
    fills: List[ExecFill] = field(default_factory=list)
    paper_exit_px: float | None = None
    real_exit_px: float | None = None

    def add_fill(self, px: float, qty: float, maker: bool, ts_ms: int):
        self.fills.append(ExecFill(px, qty, maker, ts_ms))

    @property
    def real_entry_px(self) -> float | None:
        if not self.fills:
            return None
        notional = sum(f.px * f.qty for f in self.fills)
        qty = sum(f.qty for f in self.fills)
        return notional / qty if qty else None

    @property
    def fill_qty(self) -> float:
        return sum(f.qty for f in self.fills)

    def maker_fraction(self) -> float:
        q = sum(f.qty for f in self.fills)
        if q == 0:
            return 0.0
        mq = sum(f.qty for f in self.fills if f.maker)
        return mq / q

    def slippage_bps(self) -> float | None:
        real = self.real_entry_px
        if real is None or self.paper_entry_px is None or self.paper_entry_px == 0:
            return None
        side_sign = 1 if real >= self.paper_entry_px else -1
        return (real - self.paper_entry_px) / self.paper_entry_px * 10_000 * side_sign

    def exec_alpha(self) -> float | None:
        if self.paper_exit_px is None or self.real_exit_px is None:
            return None
        # sign-adjusted improvement vs paper
        paper_pnl = (self.paper_exit_px - self.paper_entry_px)
        real_pnl = (self.real_exit_px - (self.real_entry_px or self.paper_entry_px))
        return real_pnl - paper_pnl

__all__=['ExecAttribution']
