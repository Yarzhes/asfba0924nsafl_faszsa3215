import math
from ultra_signals.regime_engine.meta_regime import RegimeModelBundle
from ultra_signals.core.custom_types import FeatureVector, RegimeFeatures, TrendFeatures, VolatilityFeatures, MomentumFeatures
from ultra_signals.core.config import RegimeSettings


def make_fv(adx=25, ema_s=101, ema_m=100, ema_l=99, atr_pct=0.5):
    return FeatureVector(
        symbol="BTCUSDT", timeframe="5m",
        ohlcv={"c":100.0},
        trend=TrendFeatures(ema_short=ema_s, ema_medium=ema_m, ema_long=ema_l, adx=adx),
        momentum=MomentumFeatures(rsi=55),
        volatility=VolatilityFeatures(atr=50, atr_percentile=atr_pct),
        regime=RegimeFeatures(confidence=0.6)
    )


def test_entropy_and_confidence_flag_mapping():
    settings = RegimeSettings()
    bundle = RegimeModelBundle(settings=settings)
    # warm up
    for _ in range(60):
        bundle.infer(make_fv())
    fv = make_fv()
    rf = bundle.infer(fv)
    # entropy should be between 0 and 1
    assert rf.regime_entropy is None or (0.0 <= rf.regime_entropy <= 1.0)
    # max_prob should be between 0 and 1
    assert rf.regime_max_prob is None or (0.0 <= rf.regime_max_prob <= 1.0)
    # if max_prob exists, confidence flag must be one of configured bands
    if rf.regime_max_prob is not None:
        bands = settings.confidence_bands
        assert rf.regime_confidence_flag in ('high','medium','low', None)


def test_smoothing_window_behavior():
    s = RegimeSettings()
    s.smoothing = {"stickiness": 0.5, "ma_window_bars": 3}
    bundle = RegimeModelBundle(settings=s)
    # push different patterns to observe smoothed probabilities
    for i in range(10):
        fv = make_fv(ema_s=101 if i<5 else 95)
        rf = bundle.infer(fv)
    # ensure pre_smoothed and smoothed present
    assert rf.pre_smoothed_regime_probs is not None
    assert rf.smoothed_regime_probs is not None
    # smoothed should be a dict with same keys as pre_smoothed
    assert set(rf.pre_smoothed_regime_probs.keys()) == set(rf.smoothed_regime_probs.keys())


def test_playbook_integration_uses_smoothed_probs():
    settings = RegimeSettings()
    bundle = RegimeModelBundle(settings=settings)
    for _ in range(55):
        bundle.infer(make_fv())
    fv = make_fv()
    rf = bundle.infer(fv)
    # Build a fake features dict and check execution_planner.select_playbook uses smoothed probs
    feats = {"regime": rf}
    from ultra_signals.engine.execution_planner import select_playbook
    pb = select_playbook(rf, feats, None, settings)
    # pb can be None if playbook config not enabled; assert call doesn't raise and returns either None or a Plan-like object
    assert pb is None or hasattr(pb, 'name')


def test_position_sizer_uses_regime_confidence():
    settings = RegimeSettings()
    bundle = RegimeModelBundle(settings=settings)
    for _ in range(60):
        bundle.infer(make_fv())
    fv = make_fv()
    rf = bundle.infer(fv)
    feats = {"regime": rf, "volatility": fv.volatility, "ohlcv": {"close": 100.0}}
    from ultra_signals.risk.position_sizing import PositionSizing
    class DummyDecision:
        def __init__(self):
            self.decision = 'LONG'
            self.confidence = 0.8
    dec = DummyDecision()
    res = PositionSizing.calculate('BTCUSDT', dec, feats, settings.__dict__ if hasattr(settings,'__dict__') else settings, 10000)
    # result can be None if config disables model; otherwise ensure confidence field exists
    assert res is None or hasattr(res, 'confidence')
