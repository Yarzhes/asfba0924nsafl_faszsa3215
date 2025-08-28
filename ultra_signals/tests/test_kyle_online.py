import time
from ultra_signals.market.kyle_online import TimeWindowAggregator, EWKyleEstimator


def test_time_window_aggregator_basic():
    tw = TimeWindowAggregator(window_s=1.0)
    now = int(time.time() * 1000)
    # add ticks across 0.5s span
    tw.add_tick(now - 800, 100.0, 10.0)
    tw.add_tick(now - 400, 100.1, -5.0)
    tw.add_tick(now - 100, 100.15, 2.0)
    dq, dp, last = tw.window_sample(now)
    assert abs(dq - 7.0) < 1e-9
    assert abs(dp - (100.15 - 100.0)) < 1e-9
    assert last == 100.15


def test_ew_estimator_converges_on_linear_relation():
    est = EWKyleEstimator(alpha=0.1)
    true_lambda = 2e-4
    # produce many samples where dp = true_lambda * dq
    for i in range(200):
        dq = (i % 10 - 5) * 50.0
        dp = true_lambda * dq
        est.add_sample(dq, dp)

    snap = est.snapshot(last_mid=100.0)
    assert abs(snap.lambda_est - true_lambda) < 5e-5
    assert 0.0 <= snap.r2 <= 1.0
