import math
from ultra_signals.market.kyle_online import EWKyleEstimator


def test_ew_kyle_estimator_happy_path():
    est = EWKyleEstimator(alpha=0.5)
    # simulate simple proportional relationship: dp = 0.01 * dq
    lam_true = 0.01
    for i in range(1, 51):
        dq = float(i)
        dp = lam_true * dq
        est.add_sample(dq, dp)
    snap = est.snapshot(last_mid=100.0, ts=0)
    assert abs(snap.lambda_est - lam_true) < 1e-3
    assert snap.r2 > 0.9


def test_ew_kyle_estimator_small_sample_edge():
    est = EWKyleEstimator(alpha=0.5)
    # single sample should not blow up; lambda_est near 0 (no variance)
    est.add_sample(0.0, 0.0)
    snap = est.snapshot(last_mid=100.0, ts=0)
    assert isinstance(snap.lambda_est, float)
    assert snap.samples >= 0
