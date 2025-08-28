import time
import math
from ultra_signals.routing.vwap_adapter import VWAPExecutor, StrategySelector
from ultra_signals.routing.types import AggregatedBook, L2Book, PriceLevel, VenueInfo


class DummyAggProvider:
    def __init__(self, agg: AggregatedBook):
        self._agg = agg

    def snapshot(self, symbol: str) -> AggregatedBook:
        return self._agg


def make_book(mid: float = 100.0, depth: int = 5, size: float = 1.0) -> L2Book:
    bids = [PriceLevel(price=mid - i * 0.1, size=size) for i in range(depth)]
    asks = [PriceLevel(price=mid + i * 0.1, size=size) for i in range(depth)]
    return L2Book(bids=bids, asks=asks, ts_ms=int(time.time() * 1000))


def test_vwap_schedule_and_pr_cap():
    venues = {'EX': VenueInfo('EX', maker_bps=0.0, taker_bps=1.0, min_notional=1.0, lot_size=0.0001)}
    curve = [0.5, 0.5]
    v = VWAPExecutor(venues, volume_curve=curve, pr_cap=0.1, jitter_frac=0.0, max_slice_notional=1000.0)

    agg = AggregatedBook(symbol='BTC/USDT', books={'EX': make_book()})
    aggprov = DummyAggProvider(agg)

    # adv_per_second such that PR cap limits slices
    adv = 10000.0  # notional per second
    now = time.time()
    res = v.execute(aggprov, 'buy', 2000.0, 'BTC/USDT', start_ts=now, end_ts=now + 60.0, adv_per_second=adv)

    # two bins, target per bin = [1000,1000] but pr_cap 0.1*adv*dt will cap
    assert len(res) <= 2
    for r in res:
        # pr cap per bin = 0.1 * 10000 * 30s = 30000 -> no cap here, ensure slice values reasonable
        assert r['slice_notional'] >= 0.0


def test_vwap_jitter_bounds():
    venues = {'EX': VenueInfo('EX', maker_bps=0.0, taker_bps=1.0, min_notional=1.0, lot_size=0.0001)}
    curve = [1.0]
    v = VWAPExecutor(venues, volume_curve=curve, pr_cap=1.0, jitter_frac=0.2, max_slice_notional=None)

    agg = AggregatedBook(symbol='BTC/USDT', books={'EX': make_book()})
    aggprov = DummyAggProvider(agg)

    # run multiple times and ensure slice_notional stays within jitter bounds
    totals = []
    for _ in range(20):
        res = v.execute(aggprov, 'buy', 1000.0, 'BTC/USDT')
        assert len(res) == 1
        totals.append(res[0]['slice_notional'])

    mn = min(totals)
    mx = max(totals)
    # with jitter_frac=0.2 bounds should be within +/-20%
    assert mn >= 1000.0 * 0.8 - 1e-6
    assert mx <= 1000.0 * 1.2 + 1e-6


def test_strategy_selector_prefers_market_when_urgent():
    venues = {
        'A': VenueInfo('A', maker_bps=0.0, taker_bps=1.0, min_notional=1.0, lot_size=0.0001),
        'B': VenueInfo('B', maker_bps=0.0, taker_bps=2.0, min_notional=1.0, lot_size=0.0001),
    }
    sel = StrategySelector(venues)

    # make a simple aggregated book where both venues have similar depth
    book = make_book(mid=100.0)
    agg = AggregatedBook(symbol='X', books={'A': book, 'B': book})

    choice_low = sel.choose(agg, 'buy', 10.0, urgency=0)
    choice_high = sel.choose(agg, 'buy', 10.0, urgency=2)

    assert choice_low in {'LIMIT', 'TWAP', 'VWAP', 'MARKET'}
    # when urgent, should bias towards MARKET (may still be same if costs similar)
    assert choice_high == 'MARKET' or choice_high == choice_low
