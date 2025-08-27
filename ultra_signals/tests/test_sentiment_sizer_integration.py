from ultra_signals.engine.sizing.advanced_sizer import AdvancedSizer

BASIC_SETTINGS = {
    'sizer': {
        'enabled': True,
        'base_risk_pct': 0.5,
        'min_risk_pct': 0.1,
        'max_risk_pct': 1.0,
    }
}

def test_advanced_sizer_applies_sentiment_modifier():
    sizer = AdvancedSizer(BASIC_SETTINGS)
    features = {
        'p_meta': 0.55,
        'atr': 100.0,
        'sentiment': {'extreme_flag_bull': 1}
    }
    res = sizer.compute('BTCUSDT','LONG',price=40000,equity=10000,features=features)
    # If sentiment mod applied risk_pct_effective should be <= base risk pct
    assert res.risk_pct_effective <= 0.5
