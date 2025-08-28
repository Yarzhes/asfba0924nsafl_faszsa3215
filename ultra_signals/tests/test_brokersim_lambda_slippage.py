from ultra_signals.sim.broker import BrokerSim, Order, FillEvent


class DummyOB:
    def __init__(self, ladder):
        self._ladder = ladder
    def ladder(self):
        return list(self._ladder)
    def best_bid(self):
        return self._ladder[0][0] if self._ladder else None
    def best_ask(self):
        return self._ladder[0][0] if self._ladder else None
    def advance(self, ms: int):
        pass


def fake_lambda_provider_factory(lam_value):
    def _provider(symbol: str):
        return lam_value
    return _provider


def test_brokersim_slippage_monotonic_with_lambda():
    # ladder: price levels (px, qty)
    ladder = [(100.0, 1000.0), (100.5, 1000.0), (101.0, 1000.0)]
    ob = DummyOB(ladder)
    cfg = {'venue_defaults': {'maker_fee_bps': -1.0, 'taker_fee_bps': 4.0}, 'venues': {'SIM': {'slippage': {'k_temp': 1.0}}}}

    # low lambda
    sim_low = BrokerSim(cfg, ob, rng_seed=1, venue='SIM', lambda_provider=fake_lambda_provider_factory(0.0001))
    order = Order(id='o1', symbol='SYM', side='BUY', type='MARKET', qty=100.0)
    fills_low = sim_low.submit_order(order)
    px_low = fills_low[0].price if fills_low else None

    # high lambda
    sim_high = BrokerSim(cfg, ob, rng_seed=1, venue='SIM', lambda_provider=fake_lambda_provider_factory(0.01))
    order2 = Order(id='o2', symbol='SYM', side='BUY', type='MARKET', qty=100.0)
    fills_high = sim_high.submit_order(order2)
    px_high = fills_high[0].price if fills_high else None

    assert px_low is not None and px_high is not None
    # With higher lambda, execution price for buy should be higher (worse)
    assert px_high >= px_low
