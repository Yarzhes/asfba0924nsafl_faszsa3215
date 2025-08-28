from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class PriceLevel:
    price: float
    size: float


@dataclass
class L2Book:
    bids: List[PriceLevel]
    asks: List[PriceLevel]
    ts_ms: int  # timestamp of snapshot


@dataclass
class VenueInfo:
    venue: str
    maker_bps: float
    taker_bps: float
    min_notional: float
    lot_size: float
    region: Optional[str] = None


@dataclass
class AggregatedBook:
    symbol: str
    books: Dict[str, L2Book]  # venue -> L2Book


@dataclass
class CostBreakdown:
    gross_price: float
    fees_bps: float
    impact_bps: float
    latency_penalty_bps: float
    total_bps: float


@dataclass
class RouterDecision:
    allocation: Dict[str, float]  # venue -> pct (0..1)
    expected_cost_bps: float
    reason: str
