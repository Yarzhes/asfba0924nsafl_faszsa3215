import time
import math
from ultra_signals.market.kyle_lambda import KyleLambdaEstimator
from ultra_signals.market.impact_adapter import ImpactAdapter


def test_lambda_on_synthetic_linear_flow():
    est = KyleLambdaEstimator(window=100, min_samples=10)
    # generate synthetic linear relation dp = 0.0001 * dq + noise
    true_lambda = 0.0001
    for i in range(50):
        dq = (i - 25) * 10.0
        dp = true_lambda * dq + (0.00001 * ((-1)**i))
        est.add_observation(dq=dq, dp=dp, mid_price=100.0, ts=int(time.time()*1000))

    snap = est.snapshot()
    assert est.samples >= 10
    # estimate should be close to true_lambda
    assert abs(snap.lambda_est - true_lambda) < 5e-5
    assert 0.0 <= snap.r2 <= 1.0


def test_impact_adapter_hysteresis():
    a = ImpactAdapter(hi_th=2.0, lo_th=1.0, base_participation=0.02)
    # normal
    h = a.decide(lambda_z=0.2)
    assert h.impact_state == 'normal'
    # cross into elevated
    h = a.decide(lambda_z=1.5)
    assert h.impact_state in ('elevated','high')
    # cross high
    h = a.decide(lambda_z=2.5)
    assert h.impact_state == 'high'
    # drop below lo_th should eventually go to normal after hysteresis steps
    h = a.decide(lambda_z=0.5)
    # still elevated or normal depending on internal state transitions; run until normal
    for _ in range(4):
        h = a.decide(lambda_z=0.2)
    assert h.impact_state == 'normal'
