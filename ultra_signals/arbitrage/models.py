from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math

# --- Core snapshot primitives -------------------------------------------------

@dataclass
class VenueQuote:
    venue: str
    symbol: str
    bid: float
    ask: float
    bid_size: float | None = None
    ask_size: float | None = None
    ts: int | None = None  # ms

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0 if self.bid and self.ask else math.nan

    @property
    def spread_bps(self) -> float | None:
        if self.bid <= 0 or self.ask <= 0:
            return None
        mid = self.mid
        if mid <= 0:
            return None
        return (self.ask - self.bid) / mid * 10_000.0

@dataclass
class VenueDepthSummary:
    venue: str
    symbol: str
    notional_bid_top5: float = 0.0
    notional_ask_top5: float = 0.0
    notional_bid_top10: float = 0.0
    notional_ask_top10: float = 0.0
    # simple slippage proxy for configured notionals (to be refined with full depth walk)
    est_slip_bps_25k_buy: float | None = None
    est_slip_bps_25k_sell: float | None = None
    # extended per-bucket slippage (buy/sell) computed from synthetic depth walk
    slippage_bps_by_notional: Dict[str, Dict[str, float]] = field(default_factory=dict)

@dataclass
class ExecutableSpread:
    symbol: str
    venue_long: str
    venue_short: str
    raw_spread_bps: float
    exec_spread_bps_by_notional: Dict[str, float]  # e.g. {"5k": 3.2, "25k": 2.1}
    exec_after_costs_bps_by_notional: Dict[str, float] | None = None  # net after fees+slippage
    half_life_sec: float | None = None  # persistence estimate

@dataclass
class FundingSnapshot:
    venue: str
    symbol: str
    current_rate_bps: float | None = None
    next_rate_est_bps: float | None = None
    hours_to_next: float | None = None

@dataclass
class BasisSnapshot:
    symbol: str
    perp_mid: float
    spot_mid: float
    basis_bps: float

@dataclass
class GeoPremiumSnapshot:
    symbol: str
    region_a: str
    region_b: str
    premium_bps: float
    z_score: float | None = None

@dataclass
class OpportunityFlag:
    code: str  # e.g. ARB_SPREAD, CARRY_POSITIVE
    score: float
    meta: Dict[str, float] = field(default_factory=dict)

@dataclass
class ArbitrageFeatureSet:
    symbol: str
    ts: int
    quotes: List[VenueQuote]
    depth: List[VenueDepthSummary]
    executable_spreads: List[ExecutableSpread] = field(default_factory=list)
    funding: List[FundingSnapshot] = field(default_factory=list)
    basis: BasisSnapshot | None = None
    geo_premium: GeoPremiumSnapshot | None = None
    flags: List[OpportunityFlag] = field(default_factory=list)
    risk_score: float | None = None

    def to_feature_dict(self) -> Dict[str, float | int | None]:
        out: Dict[str, float | int | None] = {
            "arb_symbol": self.symbol,
            "arb_ts": self.ts,
        }
        # Best raw spread among venues
        best_raw = None
        for es in self.executable_spreads:
            for k, v in es.exec_spread_bps_by_notional.items():
                out[f"arb_spread_exec_bps@{k}"] = v
            if es.exec_after_costs_bps_by_notional:
                for k, v in es.exec_after_costs_bps_by_notional.items():
                    out[f"arb_after_costs_bps@{k}"] = v
            if es.half_life_sec is not None:
                out["arb_half_life_sec"] = es.half_life_sec
            if best_raw is None or es.raw_spread_bps > best_raw:
                best_raw = es.raw_spread_bps
        if best_raw is not None:
            out["arb_spread_bps_best"] = best_raw
        if self.basis:
            out["basis_spread_bps"] = self.basis.basis_bps
        if self.geo_premium:
            out["geo_premium_bps"] = self.geo_premium.premium_bps
            if self.geo_premium.z_score is not None:
                out["geo_premium_z"] = self.geo_premium.z_score
        if self.risk_score is not None:
            out["arb_risk_score"] = self.risk_score
        # Flag booleans
        for fl in self.flags:
            out[f"flag_{fl.code.lower()}"] = fl.score
        return out
