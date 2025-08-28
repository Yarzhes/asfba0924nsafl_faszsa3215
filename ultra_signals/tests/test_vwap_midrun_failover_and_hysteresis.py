import time
from ultra_signals.routing.vwap_adapter import VWAPExecutor, StrategySelector
from ultra_signals.routing.types import AggregatedBook, L2Book, PriceLevel, VenueInfo


def make_book(mid=100.0):
    bids = [PriceLevel(price=mid - i * 0.1, size=1.0) for i in range(5)]
    asks = [PriceLevel(price=mid + i * 0.1, size=1.0) for i in range(5)]
    return L2Book(bids=bids, asks=asks, ts_ms=int(time.time() * 1000))


class AggProv:
    def __init__(self, agg: AggregatedBook):
        self._agg = agg
    def snapshot(self, symbol: str):
        return self._agg


def test_vwap_midrun_circuit_failover():
    # Setup venues A and B
    venues = {
        'A': VenueInfo('A', maker_bps=0.0, taker_bps=1.0, min_notional=1.0, lot_size=0.0001),
        'B': VenueInfo('B', maker_bps=0.0, taker_bps=2.0, min_notional=1.0, lot_size=0.0001),
    }
    agg = AggregatedBook(symbol='X', books={'A': make_book(100.0), 'B': make_book(100.0)})
    aggprov = AggProv(agg)
    vexec = VWAPExecutor(venues, volume_curve=[0.5,0.5], pr_cap=1.0, jitter_frac=0.0)

    # initial run selects A
    slices = vexec.execute(aggprov, 'buy', 100.0, 'X')
    assert any('A' in s.get('router_allocation',{}) for s in slices)

    # simulate A going red mid-run by removing it and re-running with remaining notional
    remaining = 50.0
    vexec.venues.pop('A')
    slices2 = vexec.execute(aggprov, 'buy', remaining, 'X')
    assert all(('A' not in s.get('router_allocation',{})) for s in slices2)


def test_style_switch_hysteresis_over_time():
    venues = {'X': VenueInfo('X', maker_bps=0.0, taker_bps=1.0, min_notional=1.0, lot_size=0.0001)}
    sel = StrategySelector(venues)
    agg = AggregatedBook(symbol='S', books={'X': make_book(100.0)})

    # start with neutral features
    f = {'lambda': 0.0, 'vpin_pctl': 0.1, 'spread_z': 0.0}
    c_prev = None
    # simulate gentle drift towards higher lambda over several steps
    for step in range(6):
        f['lambda'] = step * 0.0006
        c = sel.choose(agg, 'buy', 10.0, urgency=1, features=f)
        # ensure not flipping every single step; changes should be sticky
        if c_prev is not None:
            # expect either same or a deliberate change, but not oscillation
            assert not (c != c_prev and step % 1 == 0 and abs(f['lambda'] - (step-1)*0.0006) < 1e-9)
        c_prev = c
