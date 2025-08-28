from __future__ import annotations
from typing import List, Dict
from statistics import mean
import math, time, collections
from .models import (
    VenueQuote, VenueDepthSummary, FundingSnapshot, BasisSnapshot,
    GeoPremiumSnapshot, ExecutableSpread, ArbitrageFeatureSet, OpportunityFlag
)


class ArbitrageAnalyzer:
    """Enhanced analyzer with rolling z-scores, half-life, after-cost spreads."""

    def __init__(self, config: dict, venue_regions: Dict[str, str]):
        self._cfg = config or {}
        self._venue_regions = venue_regions or {}
        self._notional_buckets = [str(v) for v in (self._cfg.get('notional_buckets_usd') or [5000, 25000, 50000])]
        self._spread_history = collections.deque(maxlen=int(self._cfg.get('history_max', 500)))  # (ts,key,value)
        self._geo_history = collections.deque(maxlen=int(self._cfg.get('geo_history_max', 300)))  # (ts,premium)
        self._fee_table = (self._cfg.get('fee_overrides_bps') or {})

    def build_feature_set(
        self,
        symbol: str,
        quotes: List[VenueQuote],
        depth: List[VenueDepthSummary],
        funding: List[FundingSnapshot],
        ts: int,
    ) -> ArbitrageFeatureSet:
        depth_map = {(d.venue, d.symbol): d for d in depth}
        executable_spreads = self._compute_executable_spreads(symbol, quotes, depth_map)
        basis = self._compute_basis(symbol, quotes)
        geo = self._compute_geo_premium(symbol, quotes)
        flags: List[OpportunityFlag] = []
        risk_score = self._compute_risk_score(executable_spreads, basis, geo, funding)
        min_after_cost = float(self._cfg.get('min_after_cost_bps', 1.5))
        for es in executable_spreads:
            best_bucket_val = max(es.exec_after_costs_bps_by_notional.values()) if es.exec_after_costs_bps_by_notional else None
            if best_bucket_val and best_bucket_val >= min_after_cost:
                flags.append(OpportunityFlag(code='ARB_SPREAD', score=best_bucket_val, meta={'venue_long': es.venue_long, 'venue_short': es.venue_short}))
        if basis and abs(basis.basis_bps) >= float(self._cfg.get('basis_threshold_bps', 5.0)):
            flags.append(OpportunityFlag(code='BASIS', score=basis.basis_bps))
        if geo and abs(geo.premium_bps) >= float(self._cfg.get('geo_premium_threshold_bps', 2.0)):
            flags.append(OpportunityFlag(code='GEO_PREMIUM', score=geo.premium_bps))
        return ArbitrageFeatureSet(
            symbol=symbol,
            ts=ts,
            quotes=quotes,
            depth=depth,
            executable_spreads=executable_spreads,
            funding=funding,
            basis=basis,
            geo_premium=geo,
            flags=flags,
            risk_score=risk_score,
        )

    def _compute_executable_spreads(self, symbol: str, quotes: List[VenueQuote], depth_map: dict) -> List[ExecutableSpread]:
        spreads: List[ExecutableSpread] = []
        for i in range(len(quotes)):
            for j in range(i + 1, len(quotes)):
                a = quotes[i]; b = quotes[j]
                if a.symbol != symbol or b.symbol != symbol:
                    continue
                if any(v <= 0 for v in (a.bid, a.ask, b.bid, b.ask)):
                    continue
                for rich, cheap in ((a, b), (b, a)):
                    if rich.mid <= 0 or cheap.mid <= 0:
                        continue
                    raw_bps = (rich.mid - cheap.mid) / cheap.mid * 10_000.0
                    if raw_bps <= 0:
                        continue
                    exec_raw = (rich.bid - cheap.ask) / cheap.ask * 10_000.0
                    bucket_map: Dict[str, float] = {}
                    after_costs: Dict[str, float] = {}
                    for notional in self._notional_buckets:
                        bucket_map[notional] = exec_raw
                        after_costs[notional] = self._apply_costs(exec_raw, cheap.venue, rich.venue, depth_map, symbol, notional)
                    hl = self._estimate_half_life(symbol, rich.venue, cheap.venue, exec_raw)
                    spreads.append(ExecutableSpread(
                        symbol=symbol,
                        venue_long=cheap.venue,
                        venue_short=rich.venue,
                        raw_spread_bps=raw_bps,
                        exec_spread_bps_by_notional=bucket_map,
                        exec_after_costs_bps_by_notional=after_costs,
                        half_life_sec=hl,
                    ))
        return spreads

    def _compute_basis(self, symbol: str, quotes: List[VenueQuote]) -> BasisSnapshot | None:
        spot_mids = [q.mid for q in quotes if 'spot' in q.venue or 'coinbase' in q.venue]
        perp_mids = [q.mid for q in quotes if any(k in q.venue for k in ('perp','swap','usdm'))]
        if not spot_mids or not perp_mids:
            return None
        spot_mid = mean(spot_mids)
        perp_mid = mean(perp_mids)
        if spot_mid <= 0:
            return None
        basis_bps = (perp_mid - spot_mid) / spot_mid * 10_000.0
        return BasisSnapshot(symbol=symbol, perp_mid=perp_mid, spot_mid=spot_mid, basis_bps=basis_bps)

    def _compute_geo_premium(self, symbol: str, quotes: List[VenueQuote]) -> GeoPremiumSnapshot | None:
        baskets = self._cfg.get('geo_baskets') or {'US': ['coinbase','kraken'], 'ASIA': ['binance','bybit','okx']}
        region_mids: Dict[str, List[float]] = {}
        for q in quotes:
            venue_region = self._venue_regions.get(q.venue)
            for region, ids in baskets.items():
                if q.venue in ids or venue_region == region:
                    region_mids.setdefault(region, []).append(q.mid)
        if len(region_mids) < 2:
            return None
        regions = list(region_mids.keys())
        a, b = regions[0], regions[1]
        if not region_mids[a] or not region_mids[b]:
            return None
        mid_a = mean(region_mids[a]); mid_b = mean(region_mids[b])
        if mid_b <= 0:  # avoid div by zero
            return None
        premium_bps = (mid_a - mid_b) / mid_b * 10_000.0
        self._geo_history.append((int(time.time()*1000), premium_bps))
        z = self._rolling_z([p for _, p in self._geo_history]) if len(self._geo_history) >= 20 else 0.0
        return GeoPremiumSnapshot(symbol=symbol, region_a=a, region_b=b, premium_bps=premium_bps, z_score=z)

    def _compute_risk_score(self, spreads: List[ExecutableSpread], basis: BasisSnapshot | None, geo: GeoPremiumSnapshot | None, funding: List[FundingSnapshot]) -> float | None:
        if not spreads:
            return None
        best_after_cost = max((max((es.exec_after_costs_bps_by_notional or {}).values(), default=0.0) for es in spreads), default=0.0)
        penalties = 0.0
        if basis and abs(basis.basis_bps) < 1.0:
            penalties += 0.5
        if geo and abs(geo.premium_bps) < 1.0:
            penalties += 0.5
        soon_cut = float(self._cfg.get('risk', {}).get('funding_window_hours', 0.25))
        for f in funding:
            if f.hours_to_next is not None and f.hours_to_next < soon_cut:
                penalties += 0.5
                break
        penalties += float(self._cfg.get('risk', {}).get('latency_penalty_bps', 0.0))
        return max(0.0, best_after_cost - penalties)

    # helpers
    def _apply_costs(self, exec_bps: float, venue_buy: str, venue_sell: str, depth_map: dict, symbol: str, notional: str) -> float:
        ft = self._fee_table
        buy_fee = (ft.get(venue_buy, {}) or {}).get('taker', 0.0)
        sell_fee = (ft.get(venue_sell, {}) or {}).get('taker', 0.0)
        slip_pen = 0.0
        d_buy = depth_map.get((venue_buy, symbol))
        d_sell = depth_map.get((venue_sell, symbol))
        if d_buy and d_buy.slippage_bps_by_notional.get(notional):
            slip_pen += d_buy.slippage_bps_by_notional[notional].get('buy', 0.0)
        if d_sell and d_sell.slippage_bps_by_notional.get(notional):
            slip_pen += d_sell.slippage_bps_by_notional[notional].get('sell', 0.0)
        if slip_pen == 0.0:
            slip_pen = float(self._cfg.get('slippage_bps', 0.5))
        return exec_bps - (buy_fee + sell_fee) - slip_pen

    def _estimate_half_life(self, symbol: str, venue_rich: str, venue_cheap: str, spread_bps: float) -> float | None:
        key = f"{symbol}:{venue_rich}>{venue_cheap}"
        now_ms = int(time.time()*1000)
        self._spread_history.append((now_ms, key, spread_bps))
        series = [(ts, v) for ts, k, v in self._spread_history if k == key]
        if len(series) < 4:
            return None
        latest_ts, latest_v = series[-1]
        target = latest_v * 0.5
        for ts, v in reversed(series[:-1]):
            if v <= target:
                return max(0.1, (latest_ts - ts) / 1000.0)
        return None

    def _rolling_z(self, values: List[float]) -> float:
        if len(values) < 5:
            return 0.0
        m = sum(values) / len(values)
        var = sum((x - m) ** 2 for x in values) / max(1, len(values) - 1)
        sd = math.sqrt(var) if var > 0 else 1.0
        return (values[-1] - m) / sd
