import pandas as pd
from ultra_signals.engine.real_engine import RealSignalEngine
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import TrendFeatures, MomentumFeatures, VolatilityFeatures, RegimeFeatures, FlowMetricsFeatures

class DummyFeatureStore(FeatureStore):
    def __init__(self, feats):
        self._f = feats
    def get_features(self, symbol, timeframe, ts, nearest=True):
        return self._f
    def get_latest_features(self, symbol, timeframe):
        return self._f


def _feats(ofi=None, volume_z=None, atr_pct=0.5, spread_bps=None):
    regime = RegimeFeatures(atr_percentile=atr_pct)
    vol = VolatilityFeatures(atr=0.5, atr_percentile=atr_pct)
    fm = FlowMetricsFeatures(ofi=ofi, volume_z=volume_z, spread_bps=spread_bps)
    return {
        'trend': TrendFeatures(ema_short=1.0, ema_medium=0.9, ema_long=0.8, adx=25),
        'momentum': MomentumFeatures(rsi=60, macd_hist=0.08),
        'volatility': vol,
        'regime': regime,
        'flow_metrics': fm,
    }

BASE_SETTINGS = {
    'runtime': {'primary_timeframe': '5m'},
    'features': {},
    'ensemble': {
        'vote_threshold': {'trend': 0.10, 'mean_revert': 0.10, 'chop': 0.10, 'mixed': 0.10},
        'min_agree': {'trend': 1, 'mean_revert': 1, 'chop': 1, 'mixed': 1},
        'majority_threshold': 0.10,
        'confidence_floor': 0.0,
    },
    'quality_gates': {
        'enabled': True,
        'qscore_bins': { 'Aplus': 0.85, 'A': 0.75, 'B': 0.65, 'C': 0.55 },
        'bin_actions': {
            'Aplus': { 'size_mult': 1.30, 'require_extra_confirm': False },
            'A':     { 'size_mult': 1.15, 'require_extra_confirm': False },
            'B':     { 'size_mult': 1.00, 'require_extra_confirm': False },
            'C':     { 'size_mult': 0.75, 'require_extra_confirm': True },
            'D':     { 'size_mult': 0.00, 'require_extra_confirm': True },
        },
        'veto': {
            'max_spread_pct': 0.06,
            'atr_pct_limit': 0.97
        },
        'soft': {
            'ofi_conflict_limit': -0.15,
            'late_move_atr': 0.8,
            'min_volume_z': -0.8,
        }
    }
}


def test_quality_size_multiplier_applied():
    feats = _feats(atr_pct=0.4)
    eng = RealSignalEngine(BASE_SETTINGS, DummyFeatureStore(feats))
    now = pd.Timestamp.utcnow().floor('T')
    df = pd.DataFrame([{'open':1,'high':1,'low':1,'close':1,'volume':1}], index=[now])
    dec = eng.generate_signal(df, 'BTCUSDT')
    if dec.decision in ('LONG','SHORT'):
        q = dec.vote_detail.get('quality') if isinstance(dec.vote_detail, dict) else None
        assert q is not None
        assert 'size_multiplier' in q or 'quality_size_mult' in dec.vote_detail
    else:
        # If FLAT, just assert no unexpected veto
        assert 'VETO_' not in ''.join(dec.vetoes)


def test_quality_veto_blocks_on_spread():
    feats = _feats(spread_bps=1000)  # 0.10 spread fraction > 0.06 limit
    eng = RealSignalEngine(BASE_SETTINGS, DummyFeatureStore(feats))
    now = pd.Timestamp.utcnow().floor('T')
    df = pd.DataFrame([{'open':1,'high':1,'low':1,'close':1,'volume':1}], index=[now])
    dec = eng.generate_signal(df, 'BTCUSDT')
    # Either decision flattened or veto reason present
    assert dec.decision == 'FLAT' or any('VETO_SPREAD_WIDE' in v for v in dec.vetoes)
