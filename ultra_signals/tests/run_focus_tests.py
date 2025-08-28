import sys
import os
# Ensure project root is on sys.path for direct-run support
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ultra_signals.market.kyle_online import EWKyleEstimator
from ultra_signals.sim.broker import BrokerSim, Order


def test_ew_kyle_estimator_happy_path():
    est = EWKyleEstimator(alpha=0.5)
    lam_true = 0.01
    for i in range(1, 51):
        dq = float(i)
        dp = lam_true * dq
        est.add_sample(dq, dp)
    snap = est.snapshot(last_mid=100.0, ts=0)
    assert abs(snap.lambda_est - lam_true) < 1e-2, f"lambda mismatch {snap.lambda_est} vs {lam_true}"
    assert snap.r2 > 0.9, f"r2 too low: {snap.r2}"


def test_ew_kyle_estimator_small_sample_edge():
    est = EWKyleEstimator(alpha=0.5)
    est.add_sample(0.0, 0.0)
    snap = est.snapshot(last_mid=100.0, ts=0)
    assert isinstance(snap.lambda_est, float)
    assert snap.samples >= 0


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
    ladder = [(100.0, 1000.0), (100.5, 1000.0), (101.0, 1000.0)]
    ob = DummyOB(ladder)
    cfg = {'venue_defaults': {'maker_fee_bps': -1.0, 'taker_fee_bps': 4.0}, 'venues': {'SIM': {'slippage': {'k_temp': 1.0}}}}

    sim_low = BrokerSim(cfg, ob, rng_seed=1, venue='SIM', lambda_provider=fake_lambda_provider_factory(0.0001))
    order = Order(id='o1', symbol='SYM', side='BUY', type='MARKET', qty=100.0)
    fills_low = sim_low.submit_order(order)
    px_low = fills_low[0].price if fills_low else None

    sim_high = BrokerSim(cfg, ob, rng_seed=1, venue='SIM', lambda_provider=fake_lambda_provider_factory(0.01))
    order2 = Order(id='o2', symbol='SYM', side='BUY', type='MARKET', qty=100.0)
    fills_high = sim_high.submit_order(order2)
    px_high = fills_high[0].price if fills_high else None

    assert px_low is not None and px_high is not None
    assert px_high >= px_low, f"expected px_high >= px_low but {px_high} < {px_low}"


if __name__ == '__main__':
    tests = [
        test_ew_kyle_estimator_happy_path,
        test_ew_kyle_estimator_small_sample_edge,
        test_brokersim_slippage_monotonic_with_lambda,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -> {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {t.__name__} -> {e}")
            failed += 1
    if failed:
        print(f"{failed} test(s) failed")
        sys.exit(2)
    print("All tests passed")
    sys.exit(0)
