"""Execution adapter: sizing suggestions and style hints from L-VaR outputs."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class SizeSuggestion:
    size_multiplier: float
    exec_style: str  # MARKET | PASSIVE | TWAP
    reason: Optional[str] = None


class ExecAdapter:
    def __init__(self, liq_cost_max_pct_equity: float = 0.01, lvar_max_pct_equity: float = 0.02):
        self.liq_cost_max_pct_equity = float(liq_cost_max_pct_equity)
        self.lvar_max_pct_equity = float(lvar_max_pct_equity)

    def suggest(self, lvar_pct: float, liq_cost_pct: float, ttl_minutes: float) -> SizeSuggestion:
        # veto if liquidation cost too high vs equity
        if liq_cost_pct >= self.liq_cost_max_pct_equity:
            return SizeSuggestion(size_multiplier=0.0, exec_style='VETO', reason='LIQ_COST_EXCEEDS')
        # aggressively cut if lvar large
        if lvar_pct >= self.lvar_max_pct_equity:
            return SizeSuggestion(size_multiplier=0.5, exec_style='TWAP', reason='LVAR_HIGH')
        # if TTL long, prefer TWAP/passive
        if ttl_minutes > 60:
            return SizeSuggestion(size_multiplier=0.8, exec_style='TWAP', reason='TTL_LONG')
        # default: market
        return SizeSuggestion(size_multiplier=1.0, exec_style='MARKET', reason='NORMAL')
