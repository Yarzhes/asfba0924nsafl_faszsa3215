import math
from ultra_signals.market.kyle_online import EWKyleEstimator


def test_estimator_with_notional_vs_qty_regressor():
    # Create estimator with aggressive alpha for fast learning
    est_qty = EWKyleEstimator(alpha=0.3)
    est_notional = EWKyleEstimator(alpha=0.3)

    # Construct samples where dp = 0.0001 * notional and notional = price * qty * sign
    # For the qty-based estimator, we'll create dp proportional to qty*price_effect so it differs.
    price = 100.0
    lam_notional = 0.0001

    # Feed both estimators with same underlying trades but est_qty uses qty as x while est_notional uses signed notional
    for q in [10, -20, 30, -40, 50, -60, 70, -80, 90, -100]:
        notional = price * float(q)
        dp = lam_notional * notional
        # qty-regressor: x = q
        est_qty.add_sample(float(q), float(dp), use_notional=False)
        # notional-regressor: x = notional
        est_notional.add_sample(float(q), float(dp), use_notional=True, notional=notional)

    snap_qty = est_qty.snapshot(last_mid=price)
    snap_not = est_notional.snapshot(last_mid=price)

    # The notional-regressor estimator should recover lam_notional (within tolerance)
    assert abs(snap_not.lambda_est - lam_notional) < 1e-5
    # The qty-based estimator should not match lam_notional because scaling differs (sanity check)
    assert abs(snap_qty.lambda_est - lam_notional) > 1e-6
