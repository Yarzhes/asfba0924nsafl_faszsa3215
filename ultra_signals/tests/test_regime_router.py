from types import SimpleNamespace
from ultra_signals.engine.regime_router import RegimeRouter

class DummyTrend(SimpleNamespace):
    pass
class DummyVol(SimpleNamespace):
    pass
class DummyMom(SimpleNamespace):
    pass

SETTINGS = {
    'regime_detection': {
        'adx_threshold': 22,
        'chop_volatility': 15,
        'mean_revert_rsi': 70,
    }
}

def test_trend_detection():
    feats = {'trend': DummyTrend(adx=30), 'volatility': DummyVol(atr_percentile=50), 'momentum': DummyMom(rsi=55)}
    r = RegimeRouter.detect_regime(feats, SETTINGS)
    assert r == 'trend'

def test_chop_detection():
    feats = {'trend': DummyTrend(adx=10), 'volatility': DummyVol(atr_percentile=10), 'momentum': DummyMom(rsi=55)}
    r = RegimeRouter.detect_regime(feats, SETTINGS)
    assert r == 'chop'

def test_mean_revert_detection_high_rsi():
    feats = {'trend': DummyTrend(adx=15), 'volatility': DummyVol(atr_percentile=40), 'momentum': DummyMom(rsi=75)}
    r = RegimeRouter.detect_regime(feats, SETTINGS)
    assert r == 'mean_revert'

def test_mean_revert_detection_low_rsi():
    feats = {'trend': DummyTrend(adx=15), 'volatility': DummyVol(atr_percentile=40), 'momentum': DummyMom(rsi=25)}
    r = RegimeRouter.detect_regime(feats, SETTINGS)
    assert r == 'mean_revert'
