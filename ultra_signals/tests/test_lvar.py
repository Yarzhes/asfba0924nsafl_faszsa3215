import math
from ultra_signals.risk.lvar import LVarEngine


def test_lvar_monotonic_with_shallower_depth():
    eng = LVarEngine(equity=100000)
    # constants
    sigma = 0.02
    z = 2.33
    notional = 10000
    price = 100
    q = notional / price
    adv = 1_000_000
    pr = 0.1
    lam = 0.00001

    # deeper book -> lower liq cost
    deep_cost = eng.compute(sigma=sigma, z_alpha=z, notional=notional, price=price, q=q, adv=adv, pr=pr, book_depth=100000, lam=lam)
    shallow_cost = eng.compute(sigma=sigma, z_alpha=z, notional=notional, price=price, q=q, adv=adv, pr=pr, book_depth=1000, lam=lam)
    assert deep_cost.liq_cost_usd <= shallow_cost.liq_cost_usd


def test_ttl_increases_with_lower_pr():
    eng = LVarEngine(equity=100000)
    sigma = 0.02
    z = 2.33
    notional = 10000
    price = 100
    q = notional / price
    adv = 500_000
    book_depth = 10000
    lam = 0.00001

    high_pr = eng.compute(sigma=sigma, z_alpha=z, notional=notional, price=price, q=q, adv=adv, pr=0.2, book_depth=book_depth, lam=lam)
    low_pr = eng.compute(sigma=sigma, z_alpha=z, notional=notional, price=price, q=q, adv=adv, pr=0.01, book_depth=book_depth, lam=lam)
    assert high_pr.ttl_minutes < low_pr.ttl_minutes


def test_stress_multiplier_increases_liq_cost():
    eng = LVarEngine(equity=100000)
    sigma = 0.02
    z = 2.33
    notional = 10000
    price = 100
    q = notional / price
    adv = 1_000_000
    pr = 0.1
    lam = 0.00001

    normal = eng.compute(sigma=sigma, z_alpha=z, notional=notional, price=price, q=q, adv=adv, pr=pr, book_depth=10000, lam=lam, stress_multiplier=1.0)
    stressed = eng.compute(sigma=sigma, z_alpha=z, notional=notional, price=price, q=q, adv=adv, pr=pr, book_depth=10000, lam=lam, stress_multiplier=3.0)
    assert stressed.liq_cost_usd >= normal.liq_cost_usd


def test_lambda_extreme_increases_impact_cost():
    eng = LVarEngine(equity=100000)
    sigma = 0.02
    z = 2.33
    notional = 50000
    price = 100
    q = notional / price
    adv = 1_000_000
    pr = 0.05
    book_depth = 5000

    small_lambda = eng.compute(sigma=sigma, z_alpha=z, notional=notional, price=price, q=q, adv=adv, pr=pr, book_depth=book_depth, lam=1e-8)
    large_lambda = eng.compute(sigma=sigma, z_alpha=z, notional=notional, price=price, q=q, adv=adv, pr=pr, book_depth=book_depth, lam=1e-3)
    assert large_lambda.liq_cost_usd >= small_lambda.liq_cost_usd
