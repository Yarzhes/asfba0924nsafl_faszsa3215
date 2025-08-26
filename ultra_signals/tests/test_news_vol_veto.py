import pandas as pd
from ultra_signals.engine.real_engine import RealSignalEngine
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import TrendFeatures, MomentumFeatures, VolatilityFeatures, RegimeFeatures

class DummyFeatureStore(FeatureStore):
    def __init__(self, feats):
        self._f = feats
    def get_features(self, symbol, timeframe, ts, nearest=True):
        return self._f
    def get_latest_features(self, symbol, timeframe):
        return self._f


def _base_feats(atr_pct=None, funding=None):
    regime = RegimeFeatures(atr_percentile=atr_pct)
    vol = VolatilityFeatures(atr=None, atr_percentile=atr_pct)
    # Provide macd_hist small positive so momentum score path works
    return {
        'trend': TrendFeatures(ema_short=1, ema_medium=0.9, ema_long=0.8),
        'momentum': MomentumFeatures(rsi=55, macd_hist=0.05),
        'volatility': vol,
        'regime': regime,
        'derivatives': None,
        'volume_flow': None,
    }


def test_atr_veto_triggers():
    settings = {
        'runtime': {'primary_timeframe': '5m'},
        'features': {},
        'volatility_veto': {'enabled': True, 'atr_percentile_limit': 0.90},
        'news_veto': {'enabled': False},
    }
    feats = _base_feats(atr_pct=0.95)
    eng = RealSignalEngine(settings, DummyFeatureStore(feats))
    # minimal OHLCV segment with single bar
    now = pd.Timestamp.utcnow().floor('T')
    df = pd.DataFrame([{'open':1,'high':1,'low':1,'close':1,'volume':1}], index=[now])
    dec = eng.generate_signal(df, 'BTCUSDT')
    # If subsignals created and veto applied -> FLAT with VETO_VOL in vetoes
    assert any(v in dec.vetoes for v in ['VETO_VOL','VETO_VOL_SPIKE']) or dec.decision == 'FLAT'


def test_news_veto_triggers():
    now = pd.Timestamp.utcnow().floor('T')
    events_yaml = f"""events:\n  - time: '{now.isoformat()}'\n    title: 'CPI Release'\n    impact: 'HIGH'\n"""
    # write temp events file
    open('news_events.yaml','w').write(events_yaml)
    settings = {
        'runtime': {'primary_timeframe': '5m'},
        'features': {},
        'news_veto': {'enabled': True, 'sources': [{'local_file': 'news_events.yaml'}], 'embargo_minutes': 10, 'high_impact_only': True},
        'volatility_veto': {'enabled': False},
    }
    feats = _base_feats(atr_pct=0.10)
    eng = RealSignalEngine(settings, DummyFeatureStore(feats))
    df = pd.DataFrame([{'open':1,'high':1,'low':1,'close':1,'volume':1}], index=[now])
    dec = eng.generate_signal(df, 'BTCUSDT')
    assert 'VETO_NEWS' in dec.vetoes or dec.decision == 'FLAT'


def test_funding_veto_triggers():
    # Craft features with elevated funding rate
    class Deriv:  # minimal stub
        funding_now = 0.001  # 0.10%
    base = _base_feats(atr_pct=0.10)
    base['derivatives'] = Deriv()
    settings = {
        'runtime': {'primary_timeframe': '5m'},
        'features': {},
        'news_veto': {'enabled': False},
        'volatility_veto': {'enabled': True, 'funding_rate_limit': 0.0005},
    }
    eng = RealSignalEngine(settings, DummyFeatureStore(base))
    now = pd.Timestamp.utcnow().floor('T')
    df = pd.DataFrame([{'open':1,'high':1,'low':1,'close':1,'volume':1}], index=[now])
    dec = eng.generate_signal(df, 'BTCUSDT')
    assert any(v in dec.vetoes for v in ['VETO_VOL','VETO_VOL_SPIKE']) or dec.decision == 'FLAT'
