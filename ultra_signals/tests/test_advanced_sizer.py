import math
from ultra_signals.engine.sizing.advanced_sizer import AdvancedSizer

def _base_settings():
    return {
        'sizer': {
            'enabled': True,
            'base_risk_pct': 0.5,
            'min_risk_pct': 0.10,
            'max_risk_pct': 1.25,
            'conviction': {
                'use_meta': True,
                'use_mtc': True,
                'use_liquidity': True,
                'meta_anchor': 0.55,
                'meta_span': 0.15,
                'mtc_bonus': 1.15,
                'mtc_partial': 0.75,
                'liquidity_dampen': 0.75,
            },
            'kelly': {'enabled': True, 'cap_fraction': 0.25, 'win_R':1.0, 'loss_R':1.0},
            'dd_scaler': { 'enabled': True, 'thresholds': [ {'dd':0.05,'mult':0.75},{'dd':0.10,'mult':0.50} ]},
            'vol_target': {'method':'atr','lookback_bars':14,'target_R_multiple':1.0,'vol_floor_bps':20},
            'per_symbol': {'max_risk_pct':0.75},
            'portfolio': {'max_gross_risk_pct': 2.0, 'max_net_long_pct':3.0, 'max_net_short_pct':3.0},
            'rounding': {'step_size':0.0001},
            'safety': {'min_notional': 10}
        }
    }


def test_meta_mapping_anchor_near_one():
    s = AdvancedSizer(_base_settings())
    equity=10_000; price=100
    # p near anchor -> conv_meta ~1
    r = s.compute('BTCUSDT','LONG',price,equity,{'p_meta':0.55,'atr':1.0})
    assert abs(r.breakdown['conv_meta']-1.0) < 0.05
    # higher p increases
    r2 = s.compute('BTCUSDT','LONG',price,equity,{'p_meta':0.70,'atr':1.0})
    assert r2.breakdown['conv_meta'] > r.breakdown['conv_meta']
    # lower p decreases
    r3 = s.compute('BTCUSDT','LONG',price,equity,{'p_meta':0.40,'atr':1.0})
    assert r3.breakdown['conv_meta'] < r.breakdown['conv_meta']


def test_kelly_cap():
    s = AdvancedSizer(_base_settings())
    equity=10_000; price=100
    # p=0.5 -> edge ~0 -> kelly_mult ==1
    r = s.compute('BTCUSDT','LONG',price,equity,{'p_meta':0.50,'atr':1.0})
    assert abs(r.breakdown['kelly_mult']-1.0) < 1e-6
    # high p -> capped
    r2 = s.compute('BTCUSDT','LONG',price,equity,{'p_meta':0.95,'atr':1.0})
    cap = _base_settings()['sizer']['kelly']['cap_fraction']
    assert r2.breakdown['kelly_mult'] <= 1.0 + cap + 1e-9


def test_dd_scaling():
    cfg = _base_settings()
    s = AdvancedSizer(cfg)
    equity=10_000; price=100
    r_low = s.compute('BTCUSDT','LONG',price,equity,{'p_meta':0.55,'atr':1.0,'drawdown':0.0})
    r_dd = s.compute('BTCUSDT','LONG',price,equity,{'p_meta':0.55,'atr':1.0,'drawdown':0.08})
    assert r_dd.breakdown['dd_mult'] < r_low.breakdown['dd_mult']


def test_vol_target_inverse_relation():
    s = AdvancedSizer(_base_settings())
    equity=10_000; price=100
    # Larger ATR -> smaller qty (since stop wider)
    r_small_atr = s.compute('BTCUSDT','LONG',price,equity,{'p_meta':0.55,'atr':0.5})
    r_large_atr = s.compute('BTCUSDT','LONG',price,equity,{'p_meta':0.55,'atr':5.0})
    assert r_large_atr.qty < r_small_atr.qty


def test_symbol_and_portfolio_clamps():
    cfg = _base_settings()
    cfg['sizer']['per_symbol']['max_risk_pct'] = 0.30
    s = AdvancedSizer(cfg)
    equity=10_000; price=100
    # Artificially high meta to push risk above cap
    r = s.compute('BTCUSDT','LONG',price,equity,{'p_meta':0.95,'atr':1.0})
    assert r.clamped_by_symbol or 'PER_SYMBOL_CAP' in r.breakdown.get('reasons',[])
    # Portfolio gross: existing positions consume almost all gross risk
    # Each open position risk_amount ~ equity * risk_pct/100
    open_positions=[{'symbol':'ETHUSDT','risk_amount':equity*0.019,'side':'LONG'}]
    cfg['sizer']['portfolio']['max_gross_risk_pct']=2.0
    s2 = AdvancedSizer(cfg)
    r2 = s2.compute('BTCUSDT','LONG',price,equity,{'p_meta':0.70,'atr':1.0,'open_positions':open_positions})
    # If no room left risk_amount may be zero
    # After clamp remaining allowable gross risk may be small but non-zero due to rounding
    assert r2.risk_amount <= equity*0.0015  # effectively strongly clamped
