import pandas as pd
from ultra_signals.engine.real_engine import RealSignalEngine
from ultra_signals.core.feature_store import FeatureStore


class DummyFeatureStore(FeatureStore):
    def __init__(self):
        super().__init__(warmup_periods=2, settings={'features': {'warmup_periods': 2, 'whales': {'enabled': True}}})
        self._feat = None

    def get_features(self, symbol, timeframe, ts, nearest=False):
        return self._feat

    def get_latest_features(self, symbol, timeframe):
        return self._feat

    def set_features(self, feats):
        self._feat = feats

    def get_warmup_status(self, symbol, tf):
        return 10


def test_engine_applies_whale_boost():
    store = DummyFeatureStore()
    settings = {
        'runtime': {'primary_timeframe': '5m'},
        'features': {'warmup_periods': 2, 'whale_risk': {'enabled': True, 'composite_pressure_boost_thr': 1000, 'boost_size_mult': 2.0}},
        'position_sizing': {'enabled': False},  # keep sizing simple
        'engine': {'scoring_weights': {}, 'thresholds': {'enter': 0.1, 'exit': 0.05}, 'risk': {'max_spread_pct': {'default': 0.1}}},
    'ensemble': {'min_score': 0.0, 'majority_threshold': 0.5, 'veto_trend_flip': False, 'veto_band_pierce': False, 'min_agree': {'mixed':0}},
        'filters': {'avoid_funding_minutes': 0},
        'meta_scorer': {'enabled': False},
        'quality_gates': {'enabled': False},
        'news_veto': {'enabled': False},
        'volatility_veto': {'enabled': False},
        'trend': {}, 'momentum': {}, 'volatility': {}, 'volume_flow': {},
    }
    eng = RealSignalEngine(settings, store)
    # Provide features with whale composite above boost threshold
    store.set_features({'trend': {'ema_short':2,'ema_medium':1.5,'ema_long':1.0}, 'momentum': None, 'volatility': None, 'volume_flow': None, 'regime': None, 'whales': {'composite_pressure_score': 1500}})
    ohlcv = pd.DataFrame([{'open': 1, 'high': 1, 'low': 1, 'close': 1, 'volume': 10}], index=[pd.Timestamp.utcnow()])
    dec = eng.generate_signal(ohlcv, 'BTCUSDT')
    assert dec.vote_detail.get('whale_gate', {}).get('action') == 'BOOST'
