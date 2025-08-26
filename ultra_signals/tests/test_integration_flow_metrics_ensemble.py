import pandas as pd
from ultra_signals.engine.real_engine import RealSignalEngine
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import TrendFeatures, MomentumFeatures, VolatilityFeatures, FlowMetricsFeatures, RegimeFeatures


class DummyFeatureStore(FeatureStore):  # type: ignore[misc]
    def __init__(self, feats):
        self._f = feats
    def get_features(self, symbol, timeframe, ts, nearest=True):  # noqa: D401
        return self._f
    def get_latest_features(self, symbol, timeframe):
        return self._f


def _feats(cvd_chg=0.5, oi_rate=0.1, liq_cluster=1, depth_imbalance=0.2):
    fm = FlowMetricsFeatures(cvd_chg=cvd_chg, oi_rate=oi_rate, liq_cluster=liq_cluster, depth_imbalance=depth_imbalance)
    feats = {
        'trend': TrendFeatures(ema_short=2, ema_medium=1, ema_long=0.5, adx=25),
        'momentum': MomentumFeatures(rsi=55, macd_hist=0.05),
        'volatility': VolatilityFeatures(atr=0.5, atr_percentile=0.5),
        'regime': RegimeFeatures(),
        'flow_metrics': fm,
    }
    return feats


BASE_SETTINGS = {
    'runtime': {'primary_timeframe': '5m'},
    'features': {},
    'ensemble': {
        'vote_threshold': {'trend': 0.05, 'mean_revert': 0.05, 'chop': 0.05, 'mixed': 0.05},
        'min_agree': {'trend': 1, 'mean_revert': 1, 'chop': 1, 'mixed': 1},
        'confidence_floor': 0.0,
        'use_prob_mass': True,
    },
    # Provide explicit weights for flow metric subsignals (optional default 1.0 otherwise)
    'weights_profiles': {
        'trend': {'cvd': 1.2, 'oi_rate': 1.0, 'liquidation_pulse': 1.1, 'depth_imbalance': 0.9}
    }
}


def test_flow_metrics_subsignals_present():
    feats = _feats()
    eng = RealSignalEngine(BASE_SETTINGS, DummyFeatureStore(feats))
    now = pd.Timestamp.utcnow().floor('T')
    df = pd.DataFrame([{'open':1,'high':1,'low':1,'close':1,'volume':1}], index=[now])
    dec = eng.generate_signal(df, 'BTCUSDT')
    # Ensure at least one flow metric subsignal was generated
    strat_ids = [s.strategy_id for s in dec.subsignals]
    assert any(x in strat_ids for x in ['cvd','oi_rate','liquidation_pulse','depth_imbalance'])
    # Decision structure retains vote_detail
    assert isinstance(dec.vote_detail, dict)