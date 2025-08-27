import pandas as pd
import numpy as np

from ultra_signals.macro.engine import MacroEngine

def _mk_series(seed, n=120, drift=0.0):
    rng = np.random.default_rng(seed)
    vals = 100 + np.cumsum(rng.normal(drift, 0.3, n))
    return vals

def test_carry_unwind_flag_triggers():
    n = 150
    idx = pd.date_range('2025-01-01', periods=n, freq='5min')
    btc = pd.DataFrame({'open': _mk_series(1,n), 'high': _mk_series(2,n)+1, 'low': _mk_series(3,n)-1, 'close': _mk_series(4,n), 'volume': np.random.randint(10,100,n)}, index=idx)
    # DXY & TNX spiking upward; BTC downward returns
    dxy_close = 100 + np.linspace(0, 5, n) + np.random.normal(0,0.2,n)
    tnx_close = 4 + np.linspace(0, 0.5, n) + np.random.normal(0,0.05,n)
    externals = {
        'DX-Y.NYB': pd.DataFrame({'open':dxy_close,'high':dxy_close,'low':dxy_close,'close':dxy_close,'volume':1}, index=idx),
        '^TNX': pd.DataFrame({'open':tnx_close,'high':tnx_close,'low':tnx_close,'close':tnx_close,'volume':1}, index=idx),
    }
    settings = {"cross_asset": {"enabled": True, "correlation_windows": []}}
    eng = MacroEngine(settings)
    feats = eng.compute_features(int(idx[-1].value//1_000_000), btc, None, externals)
    assert hasattr(feats, 'carry_unwind_flag')

def test_risk_off_prob_present():
    # Minimal for regime calc: provide SPY & BTC negatively correlated
    n = 130
    idx = pd.date_range('2025-01-02', periods=n, freq='5min')
    btc_close = 100 + np.linspace(0, -3, n) + np.random.normal(0,0.2,n)
    spy_close = 400 + np.linspace(0, 4, n) + np.random.normal(0,0.5,n)
    btc = pd.DataFrame({'open':btc_close,'high':btc_close+1,'low':btc_close-1,'close':btc_close,'volume':1}, index=idx)
    spy = pd.DataFrame({'open':spy_close,'high':spy_close+1,'low':spy_close-1,'close':spy_close,'volume':1}, index=idx)
    settings = {"cross_asset": {"enabled": True, "correlation_windows": [{"label":"1d","bars":120}]}}
    eng = MacroEngine(settings)
    feats = eng.compute_features(int(idx[-1].value//1_000_000), btc, None, {'SPY': spy})
    assert feats.risk_off_prob is not None