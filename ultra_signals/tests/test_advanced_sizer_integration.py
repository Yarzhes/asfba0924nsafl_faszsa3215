import time
import numpy as np
from ultra_signals.engine.sizing.advanced_sizer import AdvancedSizer

SETTINGS_BASE = {
    'sizer': {
        'enabled': True,
        'base_risk_pct': 0.5,
        'min_risk_pct': 0.1,
        'max_risk_pct': 1.0,
        'conviction': {'use_meta': True, 'meta_anchor':0.5,'meta_span':0.15},
        'kelly': {'enabled': False},
        'dd_scaler': {'enabled': True, 'thresholds': [{'dd':0.05,'mult':0.4},{'dd':0.10,'mult':0.25}], 'recovery_steps': 2},
        'vol_target': {'method':'atr','target_R_multiple':1.0,'vol_floor_bps':10},
        'per_symbol': {'max_risk_pct':0.8},
        'portfolio': {'max_gross_risk_pct':2.0},
        'rounding': {'step_size':0.0001},
        'safety': {'min_notional': 0.0}
    }
}

def build_sizer():
    return AdvancedSizer(SETTINGS_BASE)


def test_atr_inverse_monotonic():
    s = build_sizer()
    res_low_atr = s.compute('BTC','LONG', price=50000, equity=10000, features={'atr':50,'p_meta':0.55,'drawdown':0.0,'open_positions':[]})
    res_high_atr = s.compute('BTC','LONG', price=50000, equity=10000, features={'atr':150,'p_meta':0.55,'drawdown':0.0,'open_positions':[]})
    assert res_low_atr.qty > res_high_atr.qty > 0, 'Higher ATR should reduce position size'


def test_meta_probability_positive_corr():
    s = build_sizer()
    ps = np.linspace(0.4,0.7,10)
    sizes=[]
    for p in ps:
        r = s.compute('BTC','LONG', price=30000, equity=15000, features={'atr':80,'p_meta':float(p),'drawdown':0.0,'open_positions':[]})
        sizes.append(r.risk_pct_effective)
    corr = np.corrcoef(ps, sizes)[0,1]
    assert corr > 0.2, f'Expected modest positive correlation between p_meta and risk_pct_effective, got {corr}'


def test_recovery_steps_progression():
    s = build_sizer()
    # Enter deeper drawdown -> multiplier drops
    r_dd = s.compute('BTC','LONG', price=30000, equity=10000, features={'atr':70,'p_meta':0.55,'drawdown':0.11,'open_positions':[]})
    low_risk = r_dd.risk_pct_effective
    # Improve drawdown (smaller dd) multiple times to trigger staged recovery
    r1 = s.compute('BTC','LONG', price=30000, equity=10000, features={'atr':70,'p_meta':0.55,'drawdown':0.07,'open_positions':[]})
    r2 = s.compute('BTC','LONG', price=30000, equity=10000, features={'atr':70,'p_meta':0.55,'drawdown':0.06,'open_positions':[]})
    # After two improving steps (recovery_steps=2) multiplier should have increased (risk pct higher)
    assert r2.risk_pct_effective > low_risk, 'Expected staged recovery to raise risk after sufficient improving steps'


def test_portfolio_gross_clamp():
    s = build_sizer()
    # Seed existing positions using approx 0.8% risk each to exceed 2% gross
    open_positions = [
        {'symbol':'ETH','risk_amount':10000*0.008,'side':'LONG'},
        {'symbol':'SOL','risk_amount':10000*0.008,'side':'LONG'}
    ]
    r = s.compute('BTC','LONG', price=30000, equity=10000, features={'atr':60,'p_meta':0.55,'drawdown':0.0,'open_positions':open_positions})
    assert r.qty == 0 or 'PORTFOLIO_GROSS_CAP' in r.breakdown.get('reasons', []), 'Expected portfolio gross clamp to trigger'


def test_performance_under_threshold():
    s = build_sizer()
    t0 = time.perf_counter()
    n=500
    for _ in range(n):
        s.compute('BTC','LONG', price=31000, equity=12000, features={'atr':60,'p_meta':0.55,'drawdown':0.02,'open_positions':[]})
    elapsed = (time.perf_counter()-t0)*1000.0/n  # ms per call
    assert elapsed < 2.0, f'AdvancedSizer compute too slow: {elapsed:.3f}ms'
