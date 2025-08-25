import types
from ultra_signals.risk.position_sizing import PositionSizing

class DummyVol:
    def __init__(self, atr, atr_percentile=None):
        self.atr = atr
        self.atr_percentile = atr_percentile

class DummyRegime:
    def __init__(self, profile):
        self.profile = profile

class DummyDecision:
    def __init__(self, decision, confidence):
        self.decision = decision
        self.confidence = confidence


def _features(entry=100.0, atr=1.0, atr_pct=None, regime='trend', liq_cluster=0):
    fm = types.SimpleNamespace(liq_cluster=liq_cluster)
    return {
        'ohlcv': {'close': entry},
        'volatility': DummyVol(atr=atr, atr_percentile=atr_pct),
        'regime': DummyRegime(regime),
        'flow_metrics': fm
    }

SETTINGS = {
    'risk_model': {
        'enabled': True,
        'base_risk_pct': 0.02,
        'max_leverage': 10,
        'min_confidence': 0.55,
        'regime_risk': {'trend':1.2,'mean_revert':1.0,'chop':0.5},
        'atr_multiplier_stop': 2.5,
        'atr_multiplier_tp': 3.0,
        'tighten_tp_on_liq_cluster': True,
        'liq_tp_tighten_factor': 0.7,
    }
}

def test_confidence_scaling():
    f = _features()
    low = PositionSizing.calculate('X', DummyDecision('LONG', 0.56), f, SETTINGS, 10_000)
    high = PositionSizing.calculate('X', DummyDecision('LONG', 0.90), f, SETTINGS, 10_000)
    assert high.size_quote > low.size_quote


def test_volatility_reduction():
    f_low = _features(atr=1.0, atr_pct=20)
    f_high = _features(atr=1.0, atr_pct=90)
    dec = DummyDecision('LONG', 0.80)
    low = PositionSizing.calculate('X', dec, f_low, SETTINGS, 10_000)
    high = PositionSizing.calculate('X', dec, f_high, SETTINGS, 10_000)
    assert high.size_quote < low.size_quote


def test_regime_factor():
    trend = PositionSizing.calculate('X', DummyDecision('LONG', 0.80), _features(regime='trend'), SETTINGS, 10_000)
    chop = PositionSizing.calculate('X', DummyDecision('LONG', 0.80), _features(regime='chop'), SETTINGS, 10_000)
    assert trend.size_quote > chop.size_quote


def test_stop_tp_math():
    f = _features(atr=2.0)
    dec = DummyDecision('LONG', 0.80)
    r = PositionSizing.calculate('X', dec, f, SETTINGS, 10_000)
    assert abs((r.entry_price + 2.0*3.0) - r.take_profit) < 1e-6  # ATR * tp_mult
    assert abs((r.entry_price - 2.0*2.5) - r.stop_price) < 1e-6


def test_liq_cluster_tightens_tp():
    f_normal = _features(atr=1.5, liq_cluster=0)
    f_cluster = _features(atr=1.5, liq_cluster=1)
    dec = DummyDecision('LONG', 0.90)
    normal = PositionSizing.calculate('X', dec, f_normal, SETTINGS, 10_000)
    cluster = PositionSizing.calculate('X', dec, f_cluster, SETTINGS, 10_000)
    assert cluster.take_profit < normal.take_profit
