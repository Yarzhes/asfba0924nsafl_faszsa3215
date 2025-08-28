import time
from ultra_signals.routing.vwap_adapter import VWAPExecutor
from ultra_signals.routing.twap_adapter import TWAPExecutor
from ultra_signals.routing.types import RouterDecision


class DummyRouter:
    def decide(self, agg, side, target_notional, rtt_map=None):
        return RouterDecision(allocation={'SIM': 1.0}, expected_cost_bps=0.0, reason='test')
from ultra_signals.sim.router_adapter import BrokerRouterAdapter, SimExecResult
from ultra_signals.routing.types import AggregatedBook, L2Book, PriceLevel, VenueInfo


def make_book_levels(mid=100.0, levels=20, size=10.0):
    bids = [PriceLevel(price=round(mid - i * 0.5, 2), size=size) for i in range(levels)]
    asks = [PriceLevel(price=round(mid + i * 0.5, 2), size=size) for i in range(levels)]
    return L2Book(bids=bids, asks=asks, ts_ms=int(time.time() * 1000))


class DummyAggProvider:
    def __init__(self, agg: AggregatedBook):
        self._agg = agg
    def snapshot(self, symbol: str):
        return self._agg


def test_vwap_vs_market_and_twap_sim():
    # Setup a simple simulated venue
    settings = {'broker_sim': {'rng_seed': 123, 'venue_defaults': {'maker_fee_bps': -1.0, 'taker_fee_bps': 4.0}, 'orderbook': {'levels': 20}}}
    adapter = BrokerRouterAdapter(settings)

    # venues mapping for routing tools
    venues = {'SIM': VenueInfo('SIM', maker_bps=-1.0, taker_bps=4.0, min_notional=1.0, lot_size=0.0001)}

    # aggregated book uses deep levels so market sweep cost will be worse for large sizes
    agg = AggregatedBook(symbol='X', books={'SIM': make_book_levels(mid=100.0, levels=20, size=50.0)})
    aggprov = DummyAggProvider(agg)

    # VWAP execution: moderate total notional
    vexec = VWAPExecutor(venues, volume_curve=[0.5,0.5], pr_cap=0.2, jitter_frac=0.0, max_slice_notional=None, rtt_map={'SIM':20.0})
    vexec.telemetry = None

    total = 2000.0
    # Run VWAP slices to produce slice_notional and then execute each slice as a market on the sim
    slices = vexec.execute(aggprov, 'buy', total, 'X', start_ts=time.time(), end_ts=time.time()+60, adv_per_second=1000.0)

    # Execute each slice on BrokerSim via adapter (which uses its own internal SyntheticOrderBook rebuilt from last price)
    fills = []
    for s in slices:
        res: SimExecResult = adapter.execute_fast_order(symbol='X', side='LONG', size=s['slice_notional'], price=100.0, settings=settings)
        assert res.accepted
        fills.append(res.order['price'])

    vwap_realized = sum(fills)/len(fills) if fills else None

    # Market sweep: single large market order
    mres: SimExecResult = adapter.execute_fast_order(symbol='X', side='LONG', size=total, price=100.0, settings=settings)
    assert mres.accepted
    market_price = mres.order['price']

    # TWAP: use 4 slices
    texec = TWAPExecutor(router=DummyRouter(), slices=4)
    t_slices = texec.execute(aggprov, 'buy', total, 'X')
    # simulate TWAP fills (market) same as VWAP simulation
    t_fills = []
    for t in t_slices:
        res: SimExecResult = adapter.execute_fast_order(symbol='X', side='LONG', size=t['child_notional'] if isinstance(t['child_notional'], float) else (total/4), price=100.0, settings=settings)
        assert res.accepted
        t_fills.append(res.order['price'])
    twap_realized = sum(t_fills)/len(t_fills) if t_fills else None

    # Expect VWAP to be better than a single market sweep and competitive with TWAP
    assert vwap_realized is not None and market_price is not None and twap_realized is not None
    # allow small simulator variability: VWAP should not be dramatically worse than a single market sweep
    assert vwap_realized <= market_price + 2.0
    # TWAP may be similar or slightly worse depending on sim randomness; ensure not drastically worse
    assert abs(vwap_realized - twap_realized) < 5.0
