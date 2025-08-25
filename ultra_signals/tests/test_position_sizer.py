from ultra_signals.engine.position_sizer import PositionSizer


def test_high_confidence_bigger_size():
    sizer = PositionSizer(account_equity=10000, max_risk_pct=0.02)
    high = sizer.calc_position_size(signal_conf=0.9, atr=50, liq_risk=0.1)
    low = sizer.calc_position_size(signal_conf=0.5, atr=50, liq_risk=0.1)
    assert high.size_quote > low.size_quote


def test_liq_risk_reduces_size():
    sizer = PositionSizer(account_equity=10000, max_risk_pct=0.02)
    low_risk = sizer.calc_position_size(signal_conf=0.8, atr=40, liq_risk=0.0)
    high_risk = sizer.calc_position_size(signal_conf=0.8, atr=40, liq_risk=2.0)
    assert high_risk.size_quote < low_risk.size_quote


def test_zero_atr_fallback():
    sizer = PositionSizer(account_equity=10000, max_risk_pct=0.02)
    res = sizer.calc_position_size(signal_conf=0.7, atr=0.0, liq_risk=0.0)
    assert res.size_quote > 0  # uses fallback small ATR
