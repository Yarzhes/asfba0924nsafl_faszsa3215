"""Tests for Sprint 12 labeling module.

Validates:
 - Shape & required columns
 - No leakage: last `horizon_max_bars` rows must be NaN labels
 - Basic directional logic with synthetic monotonic trends
"""
import numpy as np
import pandas as pd
from ultra_signals.analytics.labeling import compute_vol_scaled_labels


def _make_trend_df(n=120, direction=1, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(100, 110 if direction>0 else 90, n)
    noise = rng.normal(0, 0.1, n)
    close = base + noise
    high = close + 0.2
    low = close - 0.2
    open_ = close - rng.normal(0, 0.05, n)
    vol = rng.integers(500, 1000, n)
    ts = pd.to_datetime(np.arange(n), unit='m', origin='2024-01-01')
    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': vol
    }, index=ts)
    return df


def test_label_columns_and_nan_tail():
    df = _make_trend_df()
    labels = compute_vol_scaled_labels(df, horizon_min_bars=4, horizon_max_bars=8)
    for col in ['label','vol_threshold_pct','mfe_pct','mae_pct','fwd_close_ret_pct']:
        assert col in labels.columns
    # Tail region without full future window must be NaN in 'label'
    tail = labels['label'].tail(8)  # horizon_max_bars
    assert tail.isna().all()


def test_monotonic_up_trend_bias_long():
    df = _make_trend_df(direction=1)
    labels = compute_vol_scaled_labels(df, horizon_min_bars=4, horizon_max_bars=8)
    # Drop NaNs
    core = labels['label'].dropna().astype(int)
    # Expect majority LONG (+1) or NO_TRADE (0), very few -1
    neg_ratio = (core == -1).mean()
    assert neg_ratio < 0.15


def test_monotonic_down_trend_bias_short():
    df = _make_trend_df(direction=-1)
    labels = compute_vol_scaled_labels(df, horizon_min_bars=4, horizon_max_bars=8)
    core = labels['label'].dropna().astype(int)
    pos_ratio = (core == 1).mean()
    assert pos_ratio < 0.15
