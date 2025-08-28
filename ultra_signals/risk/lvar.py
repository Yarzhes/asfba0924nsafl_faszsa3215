"""Liquidity-Adjusted VaR engine.

Provides LVaR computation combining model VaR and estimated liquidation costs.
"""
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class LVarResult:
    lvar_usd: float
    lvar_pct_equity: float
    liq_cost_usd: float
    ttl_minutes: float
    stress_factor: float


class LVarEngine:
    def __init__(self, equity: float, pr_cap: float = 0.12):
        self.equity = float(equity)
        self.pr_cap = float(pr_cap)

    def base_var(self, sigma: float, z_alpha: float, notional: float) -> float:
        # simple Gaussian VaR: z * sigma * notional
        return float(abs(z_alpha) * float(sigma) * float(notional))

    def depth_walk_cost(self, price: float, book_depth: float, q: float, half_spread_bps: float = 0.0001) -> float:
        # approximate cost filling top-of-book depth and worst-case deeper levels
        if book_depth <= 0.0:
            return abs(q) * price * (half_spread_bps + 0.01)  # punitive
        # fraction of top depth
        frac = min(1.0, abs(q) / book_depth)
        # cost increases non-linearly with fraction
        slippage_pct = half_spread_bps + 0.0001 * (frac ** 1.8) * 100.0
        return abs(q) * price * slippage_pct

    def impact_cost_linear(self, lam: float, q: float) -> float:
        # C_impact ≈ 0.5 * lambda * Q^2 (lambda in price per unit volume)
        return 0.5 * abs(float(lam)) * (float(q) ** 2)

    def expected_liquidation_cost(self, price: float, q: float, book_depth: float, lam: Optional[float], half_spread_bps: float = 0.0001) -> float:
        dw = self.depth_walk_cost(price, book_depth, q, half_spread_bps)
        imp = self.impact_cost_linear(lam or 0.0, q)
        fees = abs(q) * price * 0.0004  # simple taker fee
        return dw + imp + fees

    def time_to_liquidate_minutes(self, q: float, adv: float, pr: float) -> float:
        if adv <= 0 or pr <= 0:
            return float('inf')
        daily_volume = adv
        # adv provided as notional per day; convert to per-minute
        per_min = daily_volume / (24.0 * 60.0)
        target_rate = max(1e-12, min(pr, self.pr_cap)) * daily_volume
        per_min_exec = per_min * pr
        minutes = abs(q) / max(1e-12, per_min_exec)
        return float(minutes)

    def compute(self, *, sigma: float, z_alpha: float, notional: float, price: float, q: float, adv: float, pr: float, book_depth: float, lam: Optional[float], stress_multiplier: float = 1.0) -> LVarResult:
        var = self.base_var(sigma, z_alpha, notional)
        liq = self.expected_liquidation_cost(price=price, q=q, book_depth=book_depth, lam=lam)
        liq *= float(stress_multiplier)
        lvar = var + liq
        ttl = self.time_to_liquidate_minutes(q=q, adv=adv, pr=pr)
        return LVarResult(lvar_usd=float(lvar), lvar_pct_equity=float(lvar / max(1e-12, self.equity)), liq_cost_usd=float(liq), ttl_minutes=float(ttl), stress_factor=float(stress_multiplier))
