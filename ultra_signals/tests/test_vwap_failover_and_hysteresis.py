import time
from ultra_signals.routing.vwap_adapter import VWAPExecutor, StrategySelector
from ultra_signals.routing.types import AggregatedBook, L2Book, PriceLevel, VenueInfo


def make_book(mid=100.0):
    bids = [PriceLevel(price=mid - i * 0.1, size=1.0) for i in range(5)]
    asks = [PriceLevel(price=mid + i * 0.1, size=1.0) for i in range(5)]
    return L2Book(bids=bids, asks=asks, ts_ms=int(time.time() * 1000))


class DummyAggProvider:
    def __init__(self, agg: AggregatedBook):
        self._agg = agg
    def snapshot(self, symbol: str):
        return self._agg


def test_router_failover_allocation_change():
    # Two venues: A (preferred) and B (backup)
    venues = {
        'A': VenueInfo('A', maker_bps=0.0, taker_bps=1.0, min_notional=1.0, lot_size=0.0001),
        'B': VenueInfo('B', maker_bps=0.0, taker_bps=2.0, min_notional=1.0, lot_size=0.0001),
    }
    agg = AggregatedBook(symbol='X', books={'A': make_book(100.0), 'B': make_book(100.0)})
    aggprov = DummyAggProvider(agg)

    vexec = VWAPExecutor(venues, volume_curve=[1.0], pr_cap=1.0, jitter_frac=0.0)

    # normal: should pick A
    slices = vexec.execute(aggprov, 'buy', 100.0, 'X')
    assert len(slices) == 1
    assert 'A' in slices[0]['router_allocation']

    # simulate A circuit open by removing it from venues mapping and re-run
    vexec.venues.pop('A', None)
    slices2 = vexec.execute(aggprov, 'buy', 100.0, 'X')
    assert len(slices2) == 1
    assert 'B' in slices2[0]['router_allocation']


def test_strategy_selector_hysteresis():
    venues = {'X': VenueInfo('X', maker_bps=0.0, taker_bps=1.0, min_notional=1.0, lot_size=0.0001)}
    sel = StrategySelector(venues)
    # craft a fake agg (not used directly by scores in this test)
    agg = AggregatedBook(symbol='S', books={'X': make_book(100.0)})

    # first choice
    c1 = sel.choose(agg, 'buy', 10.0, urgency=1)
    # small perturbation in features should not flip choice due to hysteresis
    c2 = sel.choose(agg, 'buy', 10.0, urgency=1, features={'vpin_pctl': 0.0})
    assert c1 == c2
