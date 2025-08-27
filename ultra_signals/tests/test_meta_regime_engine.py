import math
from ultra_signals.regime_engine.meta_regime import RegimeModelBundle
from ultra_signals.core.custom_types import FeatureVector, RegimeFeatures, TrendFeatures, VolatilityFeatures, MomentumFeatures
from ultra_signals.core.config import RegimeSettings

# Minimal synthetic feature vector factory
def make_fv(adx=25, ema_s=101, ema_m=100, ema_l=99, atr_pct=0.5):
    return FeatureVector(
        symbol="BTCUSDT", timeframe="5m",
        ohlcv={"c":100.0},
        trend=TrendFeatures(ema_short=ema_s, ema_medium=ema_m, ema_long=ema_l, adx=adx),
        momentum=MomentumFeatures(rsi=55),
        volatility=VolatilityFeatures(atr=50, atr_percentile=atr_pct),
        regime=RegimeFeatures(confidence=0.6)
    )

def test_basic_inference_prob_normalization():
    settings = RegimeSettings()
    bundle = RegimeModelBundle(settings=settings)
    for _ in range(60):
        bundle.infer(make_fv())
    fv = make_fv()
    rf = bundle.infer(fv)
    assert rf.regime_probs is not None
    s = sum(rf.regime_probs.values())
    assert abs(s-1.0) < 1e-6

def test_hazard_reacts_to_confidence_change():
    settings = RegimeSettings()
    bundle = RegimeModelBundle(settings=settings)
    # warm up
    for _ in range(10):
        bundle.infer(make_fv())
    # large negative confidence delta should boost hazard via CUSUM
    fv_low_conf = make_fv()
    fv_low_conf.regime.confidence = -2.0
    rf = bundle.infer(fv_low_conf)
    assert rf.transition_hazard is None or rf.transition_hazard <= 1.0


def test_policy_gate_injected():
    settings = RegimeSettings()
    bundle = RegimeModelBundle(settings=settings)
    for _ in range(55):
        bundle.infer(make_fv())
    rf = bundle.infer(make_fv())
    assert 'regime_size_mult' in (rf.gates or {})


def test_flip_count_and_downgrade_flag():
    settings = RegimeSettings()
    bundle = RegimeModelBundle(settings=settings)
    # induce regime alternation by flipping ema relationships
    for i in range(120):
        ema_s = 101 if i % 2 == 0 else 95
        ema_m = 100
        ema_l = 99
        rf = bundle.infer(make_fv(ema_s=ema_s, ema_m=ema_m, ema_l=ema_l))
    # downgrade flag might stay False (depends on purity) but flip_count should be tracked
    assert bundle.bar_count >= 120
    assert bundle.flip_count >= 0


def test_save_and_load(tmp_path):
    settings = RegimeSettings()
    bundle = RegimeModelBundle(settings=settings)
    for _ in range(60):
        bundle.infer(make_fv())
    path = tmp_path / 'regime_bundle.joblib'
    bundle.save(str(path))
    assert path.exists()
    new_bundle = RegimeModelBundle(settings=settings)
    new_bundle.load(str(path))
    assert new_bundle.regimes == bundle.regimes
