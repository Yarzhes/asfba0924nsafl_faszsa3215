import asyncio
import pytest
from ultra_signals.arbitrage.collector import ArbitrageCollector
from ultra_signals.arbitrage.models import VenueQuote


class DummyVenue:
    def __init__(self, vid, top=None, l2=None):
        self.id = vid
        self._top = top
        self._l2 = l2

    async def get_orderbook_top(self, symbol: str):
        class T: pass
        t = T()
        if self._top is None:
            t.bid = 0
            t.ask = 0
            t.bid_size = None
            t.ask_size = None
            t.ts = None
        else:
            t.bid, t.ask, t.bid_size, t.ask_size, t.ts = self._top
        return t

    def normalize_symbol(self, s: str) -> str:
        return s

    async def fetch_l2_orderbook(self, symbol: str, limit: int = 100):
        return self._l2


@pytest.mark.asyncio
async def test_depth_fallback_empty_book():
    venues = {'v1': DummyVenue('v1', top=(0,0,None,None,None))}
    mapper = None
    cfg = {'notional_buckets_usd': [5000]}
    collector = ArbitrageCollector(venues, mapper, cfg)
    depth = await collector.fetch_depth(['BTCUSDT'])
    assert depth, 'Expected depth summary'
    assert depth[0].slippage_bps_by_notional, 'Should have slippage map even if empty'


@pytest.mark.asyncio
async def test_depth_with_l2():
    # create a dummy L2 book with two levels
    l2 = {'bids': [(49900, 0.1), (49800, 0.5)], 'asks': [(50100, 0.2), (50200, 0.3)], 'ts': 0}
    venues = {'v1': DummyVenue('v1', top=(49999,50001,1,1,1), l2=l2)}
    collector = ArbitrageCollector(venues, None, {'notional_buckets_usd': [5000]})
    depth = await collector.fetch_depth(['BTCUSDT'])
    assert depth[0].slippage_bps_by_notional['5000']['buy'] >= 0
    assert depth[0].slippage_bps_by_notional['5000']['sell'] >= 0
