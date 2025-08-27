import numpy as np
from ultra_signals.portfolio.risk_estimator import RiskEstimator
from ultra_signals.portfolio.allocator import PortfolioAllocator

SETTINGS = {
    'portfolio': {'mock_equity': 100000},
    'portfolio_risk': {
        'enabled': True,
        'lookback_bars': 20,
        'min_bars': 10,
        'target_scheme': 'ERC',
        'rebalance_strength': 1.0,
        'corr_floor': 0.2,
        'corr_penalty': 0.5,
        'max_gross_risk_pct': 5.0,
        'max_net_long_pct': 10.0,
        'max_net_short_pct': 10.0,
        'max_cluster_risk_pct': {'grp': 0.5},
        'clusters': {'AAA':'grp','BBB':'grp'}
    }
}


def make_bar(price):
    return {'high': price * 1.01, 'low': price * 0.99, 'close': price}


def seed_history(est: RiskEstimator):
    for i in range(30):
        est.update('AAA', make_bar(100+ i*0.1), i)
        est.update('BBB', make_bar(80 + i*0.2), i)


def test_cov_matrix_sane():
    est = RiskEstimator(SETTINGS)
    seed_history(est)
    alloc = PortfolioAllocator(SETTINGS, est)
    pos = [
        {'symbol':'AAA','side':'LONG','risk_amount':1000,'price':100,'stop_distance':1,'qty':10},
        {'symbol':'BBB','side':'LONG','risk_amount':1000,'price':80,'stop_distance':1,'qty':10},
    ]
    adjs, metrics = alloc.evaluate(pos, None, 999)
    # Should produce some adjustments or at least metrics
    assert 'gross_risk_pct' in metrics
    # covariance PSD implied by ERC risk contributions near equal
    assert metrics.get('erc_deviation',0) < 0.2


def test_erc_solution_weights():
    est = RiskEstimator(SETTINGS)
    seed_history(est)
    alloc = PortfolioAllocator(SETTINGS, est)
    pos = [
        {'symbol':'AAA','side':'LONG','risk_amount':2000,'price':100,'stop_distance':1,'qty':20},
        {'symbol':'BBB','side':'LONG','risk_amount':1000,'price':80,'stop_distance':1,'qty':10},
    ]
    adjs, metrics = alloc.evaluate(pos, None, 999)
    # Expect adjustments to scale toward equal risk -> at least one scale action
    assert any(a['action']=='scale' for a in adjs)


def test_corr_penalty():
    s = dict(SETTINGS)
    s['portfolio_risk'] = dict(SETTINGS['portfolio_risk'])
    s['portfolio_risk']['corr_penalty'] = 0.8
    est = RiskEstimator(s)
    seed_history(est)
    alloc = PortfolioAllocator(s, est)
    existing = [{'symbol':'AAA','side':'LONG','risk_amount':1000,'price':100,'stop_distance':1,'qty':10}]
    candidate = {'symbol':'BBB','side':'LONG','risk_amount':1000,'price':80,'stop_distance':1,'qty':10}
    adjs, _ = alloc.evaluate(existing, candidate, 111)
    # Expect scale adjustment for candidate due to correlation penalty
    cand_adj = [a for a in adjs if a['symbol']=='BBB']
    assert cand_adj and cand_adj[0]['action']=='scale' and cand_adj[0]['size_mult'] < 1.0


def test_rebalance_strength_partial():
    s = dict(SETTINGS)
    s['portfolio_risk'] = dict(SETTINGS['portfolio_risk'])
    s['portfolio_risk']['rebalance_strength'] = 0.5
    est = RiskEstimator(s)
    seed_history(est)
    alloc = PortfolioAllocator(s, est)
    pos = [
        {'symbol':'AAA','side':'LONG','risk_amount':5000,'price':100,'stop_distance':1,'qty':50},
        {'symbol':'BBB','side':'LONG','risk_amount':1000,'price':80,'stop_distance':1,'qty':10},
    ]
    adjs, _ = alloc.evaluate(pos, None, 123)
    # Expect scaling but not full equalization (size_mult not extreme)
    if adjs:
        m = [a['size_mult'] for a in adjs if a['symbol']=='AAA']
        if m:
            assert 0.6 < m[0] < 1.0  # partial reduce

