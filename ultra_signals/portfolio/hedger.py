"""Dynamic beta hedger placing leader hedge to keep portfolio beta inside band."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import math


@dataclass
class HedgePlan:
    action: str  # NONE / OPEN / ADJUST / CLOSE
    target_notional: float
    delta_notional: float
    reason: str
    veto_reason: Optional[str] = None


@dataclass
class BetaHedger:
    leader: str
    beta_band: Tuple[float, float]
    min_rebalance_frac: float
    taker_fee: float = 0.0004
    funding_penalty_perc_per_day: float = 0.03
    cooloff_bars: int = 3

    last_rebalance_bar: Optional[int] = None
    current_hedge_notional: float = 0.0  # signed (long positive)

    def compute_plan(
        self,
        bar_index: int,
        portfolio_beta: float,
        equity: float,
        beta_target: float = 0.0,
        est_slippage_bps: float = 1.0,
    ) -> HedgePlan:
        beta_min, beta_max = self.beta_band
        # inside band => maybe unwind existing hedge
        if beta_min <= portfolio_beta <= beta_max:
            if self.current_hedge_notional != 0.0:
                return HedgePlan(action="CLOSE", target_notional=0.0, delta_notional=-self.current_hedge_notional, reason="IN_BAND")
            return HedgePlan(action="NONE", target_notional=self.current_hedge_notional, delta_notional=0.0, reason="IN_BAND")

        # outside band -> compute required notional shift to bring to beta_target
        desired_notional = (portfolio_beta - beta_target) * equity
        # sign: if beta > band (positive) we short leader (negative notional) to offset
        # For simplicity store hedge notional with same sign as position on leader (long positive)
        target_notional = -desired_notional  # invert to hedge
        delta = target_notional - self.current_hedge_notional

        if equity > 0 and abs(delta) < self.min_rebalance_frac * equity:
            return HedgePlan(action="NONE", target_notional=self.current_hedge_notional, delta_notional=0.0, reason="DELTA_TOO_SMALL", veto_reason="HEDGE_COOLDOWN")

        if self.last_rebalance_bar is not None and (bar_index - self.last_rebalance_bar) < self.cooloff_bars:
            return HedgePlan(action="NONE", target_notional=self.current_hedge_notional, delta_notional=0.0, reason="COOLOFF", veto_reason="HEDGE_COOLDOWN")

        # cost heuristic (fee + slippage) vs reduction in beta overshoot
        fee_cost = abs(delta) * self.taker_fee
        slippage_cost = abs(delta) * (est_slippage_bps / 10_000.0)
        # expected benefit: reduction in |beta - target| * equity
        current_gap = abs(portfolio_beta - beta_target)
        post_gap = abs((portfolio_beta - (delta / equity)))  # approximate
        benefit = (current_gap - post_gap) * equity
        # Only block if benefit is trivially small relative to costs (<1.2x)
        if benefit > 0 and benefit < 1.2 * (fee_cost + slippage_cost):
            return HedgePlan(action="NONE", target_notional=self.current_hedge_notional, delta_notional=0.0, reason="COSTLY", veto_reason="HEDGE_COSTLY")

        return HedgePlan(action="ADJUST" if self.current_hedge_notional != 0 else "OPEN", target_notional=target_notional, delta_notional=delta, reason="OUT_OF_BAND")

    def apply_plan(self, plan: HedgePlan, bar_index: int) -> None:
        if plan.action in ("OPEN", "ADJUST", "CLOSE"):
            self.current_hedge_notional = plan.target_notional
            self.last_rebalance_bar = bar_index

    # --- NEW: helper for unrealized PnL (simplified linear model) ---
    def unrealized_pnl(self, current_price: float, avg_price: float) -> float:
        """Approximate PnL of hedge given current and average fill price.

        Assumes notional = qty * entry_price so qty = notional / avg_price.
        PnL = (current_price - avg_price) * qty with sign governed by notional sign.
        """
        try:
            if avg_price <= 0:
                return 0.0
            qty = self.current_hedge_notional / avg_price
            return (current_price - avg_price) * qty
        except Exception:
            return 0.0
