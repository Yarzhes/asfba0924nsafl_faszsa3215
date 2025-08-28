from __future__ import annotations

"""Minimal ArbitrageCollector implementation used in tests.

Provides only the small subset required by unit tests:
- fetch_quotes: gather top-of-book from venues
- fetch_depth: always produce slippage_bps_by_notional keys for configured buckets
- fetch_funding: simple history -> FundingSnapshot mapping
"""

import asyncio
import time
from typing import List, Dict, Any
from loguru import logger
from .models import VenueQuote, VenueDepthSummary, FundingSnapshot
from ultra_signals.data.funding_provider import FundingProvider


class ArbitrageCollector:
    def __init__(self, venues: Dict[str, Any], symbol_mapper, config: dict, funding_provider: FundingProvider | None = None):
        self._venues = venues
        self._mapper = symbol_mapper
        self._config = config or {}
        self._funding_provider = funding_provider

    async def fetch_quotes(self, symbols: List[str]) -> List[VenueQuote]:
        tasks = [self._fetch_quote_one(vid, v, sym) for vid, v in self._venues.items() for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: List[VenueQuote] = []
        for r in results:
            if isinstance(r, VenueQuote):
                out.append(r)
            elif isinstance(r, Exception):
                logger.debug("quote fetch error: {}", r)
        return out

    async def _fetch_quote_one(self, venue_id: str, venue, internal_symbol: str):
        try:
            venue_symbol = venue.normalize_symbol(internal_symbol)
            top = await venue.get_orderbook_top(venue_symbol)
            return VenueQuote(
                venue=venue_id,
                symbol=internal_symbol,
                bid=top.bid,
                ask=top.ask,
                bid_size=top.bid_size,
                ask_size=top.ask_size,
                ts=top.ts,
            )
        except Exception as e:  # pragma: no cover
            logger.debug("_fetch_quote_one fail {} {}: {}", venue_id, internal_symbol, e)
            return None

    async def fetch_depth(self, symbols: List[str]) -> List[VenueDepthSummary]:
        quotes = await self.fetch_quotes(symbols)
        buckets = [str(v) for v in (self._config.get('notional_buckets_usd') or [5000, 25000, 50000])]

        def _simple_slip_from_top(mid: float, size_bid, size_ask, target: float):
            # crude fallback: if top-of-book depth covers target assume small slip,
            # otherwise return None meaning unknown
            if mid and size_ask and size_bid:
                if size_ask * mid >= target and size_bid * mid >= target:
                    return 0.5, 0.5
                return None, None
            return None, None

        out: List[VenueDepthSummary] = []
        for q in quotes:
            mid = (q.bid + q.ask) / 2 if (q.bid and q.ask) else None
            slippage_map: Dict[str, Dict[str, float | None]] = {}
            for notional in buckets:
                try:
                    target = float(notional)
                except Exception:
                    target = 0.0
                if target <= 0:
                    slippage_map[notional] = {'buy': None, 'sell': None}
                    continue
                buy_slip, sell_slip = _simple_slip_from_top(mid, q.bid_size, q.ask_size, target)
                slippage_map[notional] = {'buy': buy_slip, 'sell': sell_slip}

            out.append(VenueDepthSummary(
                venue=q.venue,
                symbol=q.symbol,
                notional_bid_top5=(q.bid or 0) * (q.bid_size or 0),
                notional_ask_top5=(q.ask or 0) * (q.ask_size or 0),
                notional_bid_top10=(q.bid or 0) * (q.bid_size or 0),
                notional_ask_top10=(q.ask or 0) * (q.ask_size or 0),
                est_slip_bps_25k_buy=slippage_map.get('25000', {}).get('buy') if '25000' in slippage_map else None,
                est_slip_bps_25k_sell=slippage_map.get('25000', {}).get('sell') if '25000' in slippage_map else None,
                slippage_bps_by_notional=slippage_map,
            ))

        return out

    async def fetch_funding(self, symbols: List[str]) -> List[FundingSnapshot]:
        snaps: List[FundingSnapshot] = []
        if not self._funding_provider:
            return snaps
        for sym in symbols:
            hist = self._funding_provider.get_history(sym)
            if not hist:
                continue
            latest = hist[-1]
            rate = float(latest.get('funding_rate', 0.0) * 10_000.0)
            ft = int(latest.get('funding_time', 0))
            if ft > 0:
                now_ms = int(time.time() * 1000)
                ms_into_cycle = (now_ms - ft) % (8 * 60 * 60 * 1000)
                hours_to_next = (8 * 60 * 60 * 1000 - ms_into_cycle) / 3600_000.0
            else:
                hours_to_next = None
            snaps.append(FundingSnapshot(venue='binance', symbol=sym, current_rate_bps=rate, hours_to_next=hours_to_next))
        return snaps

    async def collect_all(self, symbols: List[str]):
        quotes = await self.fetch_quotes(symbols)
        depth = await self.fetch_depth(symbols)
        funding = await self.fetch_funding(symbols)
        return {
            'quotes': quotes,
            'depth': depth,
            'funding': funding,
            'ts': int(time.time() * 1000),
        }
