from typing import Dict
from .types import CostBreakdown, L2Book, VenueInfo


def estimate_all_in_cost(
    book: L2Book,
    side: str,
    venue: VenueInfo,
    target_notional: float,
    rtt_ms: float = 20.0,
    impact_lambda: float = 0.0,
) -> CostBreakdown:
    """Estimate gross price and bps costs for filling target_notional on venue.

    This is intentionally simple: gross_price from depth, fees from venue,
    impact modeled as impact_lambda * sqrt(notional) (toy), latency penalty
    proportional to rtt_ms.
    """
    # compute gross price (VWAP)
    from .aggregator import Aggregator

    gross = Aggregator.depth_cost(book, side, target_notional)

    if gross == float('inf'):
        return CostBreakdown(gross_price=gross, fees_bps=float('inf'), impact_bps=float('inf'), latency_penalty_bps=float('inf'), total_bps=float('inf'))

    # fee in bps
    fees = venue.taker_bps if side == 'buy' else venue.taker_bps

    # impact: toy model (bps)
    impact = impact_lambda * (target_notional ** 0.5)

    # latency penalty: 0.01 bps per ms
    latency_penalty = 0.01 * max(0.0, rtt_ms - 5.0)

    total = fees + impact + latency_penalty

    return CostBreakdown(
        gross_price=gross,
        fees_bps=fees,
        impact_bps=impact,
        latency_penalty_bps=latency_penalty,
        total_bps=total,
    )
