"""Basic tests for Sprint 42 MacroEngine scaffold.

Ensures the engine produces a MacroFeatures object with expected optional
keys when provided with synthetic BTC / SPY data. This guards against
accidental import/type regressions.
"""
import pandas as pd
import numpy as np

from ultra_signals.macro.engine import MacroEngine

def _mk_df(n=120, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    price = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    return pd.DataFrame({"open": price, "high": price+1, "low": price-1, "close": price, "volume": rng.integers(10,100,size=n)}, index=idx)

def test_macro_engine_minimal_corr():
    settings = {"cross_asset": {"enabled": True, "correlation_windows": [{"label": "30m", "bars": 30}, {"label": "1d", "bars": 120}]}}
    eng = MacroEngine(settings)
    btc = _mk_df(130, seed=42)
    spy = _mk_df(130, seed=99)
    feats = eng.compute_features(int(btc.index[-1].value // 1_000_000), btc, None, {"SPY": spy})
    # Keys are optional but correlation for btc_spy_corr_30m should be numeric
    assert hasattr(feats, 'btc_spy_corr_30m')
    # Ensure no exception path resulted in missing attribute
    assert isinstance(feats.btc_vix_proxy, (float, type(None)))