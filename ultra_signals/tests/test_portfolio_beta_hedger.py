import math
import pandas as pd
from ultra_signals.portfolio.correlations import RollingCorrelationBeta
from ultra_signals.portfolio.hedger import BetaHedger
from ultra_signals.portfolio.risk_caps import PortfolioRiskCaps


def test_beta_estimation_stable():
    rc = RollingCorrelationBeta(leader="BTC", lookback=20)
    # create synthetic: ETH = 0.5 * BTC + noise, SOL uncorrelated
    for i in range(60):
        btc = 10000 + i * 10
        # Set ETH so its percentage return is ~0.5 * BTC's.
        # BTC return per step ~10/price; choose ETH increment to be ~0.5 * that * ETH price.
        eth_price = 500 + i * 0.25  # much smaller relative increment
        eth = eth_price
        sol = 50 + (i % 7)
        rc.update_price("BTC", btc, i)
        rc.update_price("ETH", eth, i)
        rc.update_price("SOL", sol, i)
    rc.recompute()
    beta_eth = rc.get_beta("ETH")
    assert 0.2 < beta_eth < 0.8, beta_eth


def test_preview_beta_guard():
    rc = RollingCorrelationBeta(leader="BTC", lookback=10)
    for i in range(15):
        px = 100 + i
        rc.update_price("BTC", px, i)
        rc.update_price("ALT", px * 1.02, i)
    rc.recompute()
    betas = rc.betas
    caps = PortfolioRiskCaps(beta_band=(-0.15, 0.15), beta_hard_cap=0.25)
    exposure = {"BTC": 0.0}
    preview = caps.preview_beta_after_trade(
        symbol="ALT",
        add_notional=1000,
        equity=10_000,
        exposure_symbols=exposure,
        betas=betas,
        cluster_map={"ALT": "majors"},
    )
    assert preview.allowed
    # large notional to breach hard cap
    preview2 = caps.preview_beta_after_trade(
        symbol="ALT",
        add_notional=100_000,
        equity=10_000,
        exposure_symbols=exposure,
        betas=betas,
        cluster_map={"ALT": "majors"},
    )
    assert not preview2.allowed and preview2.veto_reason == "BETA_CAP"


def test_hedge_sizing_in_band():
    hedger = BetaHedger(leader="BTC", beta_band=(-0.15, 0.15), min_rebalance_frac=0.005)
    plan = hedger.compute_plan(bar_index=10, portfolio_beta=0.30, equity=100_000, beta_target=0.0)
    assert plan.action in ("OPEN", "ADJUST")
    hedger.apply_plan(plan, bar_index=10)
    # after applying if we are back in band with next call we should get CLOSE or NONE
    plan2 = hedger.compute_plan(bar_index=11, portfolio_beta=0.0, equity=100_000, beta_target=0.0)
    assert plan2.action in ("NONE", "CLOSE")


def test_cost_gate_blocks_tiny_adjustments():
    hedger = BetaHedger(leader="BTC", beta_band=(-0.15, 0.15), min_rebalance_frac=0.02)  # 2% min
    plan = hedger.compute_plan(bar_index=5, portfolio_beta=0.20, equity=100_000, beta_target=0.0)
    assert plan.action in ("OPEN", "ADJUST")
    hedger.apply_plan(plan, bar_index=5)
    # Small change in beta => desired notional shift 1k < 2k threshold => NONE
    plan_small = hedger.compute_plan(bar_index=6, portfolio_beta=0.19, equity=100_000, beta_target=0.0)
    assert plan_small.action == "NONE"
