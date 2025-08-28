"""Integration-style test skeleton: assert BrokerSim slippage scales with injected lambda.

This is a lightweight test that constructs a BrokerSim with a SyntheticOrderBook and a
lambda provider that returns fixed values per symbol. It submits identical market orders
across symbols with different lambda and checks that execution price shift correlates.

Note: this test is a scaffold and may need adjustment to fit CI timing assumptions.
"""
from ultra_signals.sim.orderbook import SyntheticOrderBook
from ultra_signals.sim.broker import BrokerSim, Order


def test_brokersim_impact_correlates_with_lambda():
    # fixed settings, small ladder
    settings = {
        'venue_defaults': {'maker_fee_bps': -1.0, 'taker_fee_bps': 4.0},
        'venues': {'SIM': {'slippage': {'k_temp': 1.0, 'impact_factor': 0.5}, 'latency_ms': {}}}
    }
    ob = SyntheticOrderBook('TEST', levels=5, seed=123, base_spread_bps=2.0)
    ob.rebuild_from_bar({'close': 100.0, 'high': 101.0, 'low': 99.0})

    # provider that maps symbol->lambda
    lam_map = {'L1': 1e-5, 'L2': 5e-5, 'L3': 2e-4}
    def provider(sym):
        return lam_map.get(sym, 0.0)

    sim_L1 = BrokerSim(settings, ob, rng_seed=1, venue='SIM', lambda_provider=lambda s: provider('L1'))
    sim_L2 = BrokerSim(settings, ob, rng_seed=1, venue='SIM', lambda_provider=lambda s: provider('L2'))
    sim_L3 = BrokerSim(settings, ob, rng_seed=1, venue='SIM', lambda_provider=lambda s: provider('L3'))

    # submit same qty order to each sim
    ord1 = Order(id='o1', symbol='L1', side='BUY', type='MARKET', qty=10.0)
    ord2 = Order(id='o2', symbol='L2', side='BUY', type='MARKET', qty=10.0)
    ord3 = Order(id='o3', symbol='L3', side='BUY', type='MARKET', qty=10.0)

    f1 = sim_L1.submit_order(ord1)[0]
    f2 = sim_L2.submit_order(ord2)[0]
    f3 = sim_L3.submit_order(ord3)[0]

    # Expect exec price increases with lambda (approx)
    assert f1.price <= f2.price <= f3.price
