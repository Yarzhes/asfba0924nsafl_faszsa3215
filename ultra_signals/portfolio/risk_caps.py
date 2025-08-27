"""Correlation & beta aware exposure / cluster caps before trade open."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class BetaPreview:
    allowed: bool
    scaled_notional: float
    veto_reason: Optional[str]
    projected_beta: float


@dataclass
class PortfolioRiskCaps:
    beta_band: Tuple[float, float]
    beta_hard_cap: float
    block_if_exceeds_beta_cap: bool = True
    downscale_if_over_band: bool = True
    downscale_factor: float = 0.5
    cluster_caps: Dict[str, float] = None  # fraction of equity

    def preview_beta_after_trade(
        self,
        symbol: str,
        add_notional: float,
        equity: float,
        exposure_symbols: Dict[str, float],
        betas: Dict[str, float],
        cluster_map: Dict[str, str],
    ) -> BetaPreview:
        # project symbol notionals
        projected = dict(exposure_symbols)
        projected[symbol] = projected.get(symbol, 0.0) + float(add_notional)
        # compute beta
        if equity <= 0:
            return BetaPreview(True, add_notional, None, 0.0)
        beta_p = 0.0
        for sym, notional in projected.items():
            beta = float(betas.get(sym, 0.0))
            beta_p += (notional / equity) * beta
        beta_min, beta_max = self.beta_band
        # Hard cap check
        if abs(beta_p) > self.beta_hard_cap and self.block_if_exceeds_beta_cap:
            return BetaPreview(False, 0.0, "BETA_CAP", beta_p)
        # Band downscale
        if self.downscale_if_over_band and (beta_p < beta_min or beta_p > beta_max):
            return BetaPreview(True, add_notional * self.downscale_factor, None, beta_p)
        return BetaPreview(True, add_notional, None, beta_p)

    def check_cluster_caps(
        self,
        symbol: str,
        add_notional: float,
        equity: float,
        exposure_symbols: Dict[str, float],
        cluster_map: Dict[str, str],
    ) -> Optional[str]:
        if not self.cluster_caps or equity <= 0:
            return None
        cl = cluster_map.get(symbol)
        if not cl:
            return None
        curr = 0.0
        for sym, notional in exposure_symbols.items():
            if cluster_map.get(sym) == cl:
                curr += abs(notional)
        new_total = curr + abs(add_notional)
        cap_notional = self.cluster_caps.get(cl, 1e12) * equity
        if new_total > cap_notional:
            return "CLUSTER_CAP"
        return None
