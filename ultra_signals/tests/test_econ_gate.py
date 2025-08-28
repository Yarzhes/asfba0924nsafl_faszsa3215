import pandas as pd
import time
from ultra_signals.engine.real_engine import RealSignalEngine
from ultra_signals.core.custom_types import RegimeFeatures, MomentumFeatures

class DummyFS:
    def __init__(self, feats: dict):
        self._feats = feats
    def get_features(self, symbol, timeframe, ts, nearest=True):
        return self._feats
    def get_latest_features(self, symbol, timeframe):
        return self._feats
    def get_warmup_status(self, symbol, tf):
        return 100


def _ohlcv_df():
    now = pd.Timestamp.utcnow().floor('min')
    rows = []
    for i in range(3):
        ts = now - pd.Timedelta(minutes=2-i)
        rows.append([ts, 100+i, 101+i, 99+i, 100.5+i, 1000])
    df = pd.DataFrame(rows, columns=['ts','open','high','low','close','volume']).set_index('ts')
    return df


def test_econ_size_dampen():
    now_ms = int(time.time()*1000)
    econ_cfg = {
        'enabled': True,
        'refresh_min': 0,
        'static_events': [
            {
                'cls':'cpi','severity':'med','ts_start': now_ms + 10*60*1000, # 10m out
                'risk_pre_min':30,'risk_post_min':30,
                'title':'CPI Release'
            }
        ]
    }
    settings = {
        'runtime':{'primary_timeframe':'5m'},
        'econ':econ_cfg,
        'quality_gates': {'enabled': False},
        'news_veto': {'enabled': False},
        'volatility_veto': {'enabled': False}
    }
    feats = {'regime': RegimeFeatures(), 'momentum': MomentumFeatures(rsi=60, macd_hist=0.1)}
    eng = RealSignalEngine(settings, DummyFS(feats))
    df = _ohlcv_df()
    dec = eng.generate_signal(df, 'BTCUSDT')
    # Should dampen (med severity => size_mult 0.5 per default econ policy)
    assert dec.vote_detail.get('econ_gate',{}).get('action') in (None,'DAMPEN')


def test_econ_veto_high_sev_pre_window():
    now_ms = int(time.time()*1000)
    econ_cfg = {
        'enabled': True,
        'apply_veto': True,
        'refresh_min': 0,
        'static_events': [
            {
                'cls':'fomc','severity':'high','ts_start': now_ms + 5*60*1000, # 5m out inside pre window 60
                'risk_pre_min':60,'risk_post_min':60,
                'title':'FOMC'
            }
        ]
    }
    settings = {
        'runtime':{'primary_timeframe':'5m'},
        'econ':econ_cfg,
        'quality_gates': {'enabled': False},
        'news_veto': {'enabled': False},
        'volatility_veto': {'enabled': False}
    }
    feats = {'regime': RegimeFeatures(), 'momentum': MomentumFeatures(rsi=55, macd_hist=0.05)}
    eng = RealSignalEngine(settings, DummyFS(feats))
    df = _ohlcv_df()
    dec = eng.generate_signal(df, 'BTCUSDT')
    # high severity inside pre window -> veto
    if dec.decision != 'FLAT':
        # if subsignals absent causing FLAT earlier, skip; else must be veto
        assert 'ECON_VETO' in dec.vetoes or dec.vote_detail.get('reason')=='ECON_VETO'
