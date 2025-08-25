"""Tests for Sprint 11 Alpha V2 feature pack."""
import pandas as pd
import numpy as np
from ultra_signals.features.alpha_v2 import compute_alpha_v2_features


def _mock_ohlcv(rows: int = 60):
    base_ts = pd.Timestamp('2023-01-01 00:00:00')
    idx = [base_ts + pd.Timedelta(minutes=i) for i in range(rows)]
    data = {
        'open': np.linspace(100, 110, rows),
        'high': np.linspace(101, 111, rows) + np.random.rand(rows)*0.5,
        'low': np.linspace(99, 109, rows) - np.random.rand(rows)*0.5,
        'close': np.linspace(100, 110, rows) + np.sin(np.linspace(0, 6.28, rows))*0.5,
        'volume': np.random.randint(50,150,size=rows)
    }
    return pd.DataFrame(data, index=idx)


def test_alpha_v2_basic_keys():
    df = _mock_ohlcv()
    feats = compute_alpha_v2_features(df)
    # Required core keys
    for k in ['hh_break_20','ll_break_20','sess_vwap','sess_vwap_dev','bb_kc_ratio','volume_burst','hour','session','week_of_month']:
        assert k in feats


def test_alpha_v2_divergence_flags():
    df = _mock_ohlcv()
    feats = compute_alpha_v2_features(df)
    assert 'bull_div' in feats and 'bear_div' in feats
    assert feats['bull_div'] in (0,1)
    assert feats['bear_div'] in (0,1)


def test_alpha_v2_attribution_sum():
    df = _mock_ohlcv()
    feats = compute_alpha_v2_features(df)
    if 'attribution' in feats and feats['attribution']:
        total = sum(feats['attribution'].values())
        assert 0.99 <= total <= 1.01
