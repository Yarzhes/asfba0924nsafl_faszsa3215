"""
Tests for Volume Flow and VWAP Features
"""

import pandas as pd
import numpy as np
import pytest

from ultra_signals.features.volume_flow import (
    compute_vwap,
    compute_volume_zscore,
    compute_volume_flow_features,
)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Provides a sample OHLCV DataFrame for testing."""
    data = {
        "timestamp": pd.to_datetime(np.arange(10) * 60, unit="s"),
        "high": [102, 103, 105, 106, 104, 107, 108, 109, 110, 112],
        "low": [100, 101, 102, 103, 102, 105, 106, 107, 108, 110],
        "close": [101, 102, 104, 105, 103, 106, 107, 108, 109, 111],
        "volume": [100, 150, 200, 250, 180, 300, 350, 400, 450, 500],
    }
    df = pd.DataFrame(data)
    df = df.set_index("timestamp")
    return df


def test_compute_vwap_with_known_values(sample_ohlcv_data):
    """
    Tests the VWAP calculation against a manually computed value.
    """
    ohlcv = sample_ohlcv_data.head(5)
    typical_price = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3
    
    # Manual calculation for the 5th period (window=5)
    # TP = [101.0, 102.0, 103.66, 104.66, 103.0]
    # Vol = [100, 150, 200, 250, 180]
    # Sum(TP * Vol) = 10100 + 15300 + 20733.33 + 26166.66 + 18540 = 90840
    # Sum(Vol) = 100 + 150 + 200 + 250 + 180 = 880
    # Expected VWAP = 90840 / 880 = 103.227
    
    vwap_features = compute_vwap(typical_price, ohlcv["volume"], window=5, std_devs=(1.0,))
    
    assert "vwap" in vwap_features.columns
    assert "vwap_upper_1.0std" in vwap_features.columns
    assert "vwap_lower_1.0std" in vwap_features.columns
    
    latest_vwap = vwap_features["vwap"].iloc[-1]
    
    assert pytest.approx(103.227, abs=1e-3) == latest_vwap

    # Test that bands are scaled correctly and are not NaN
    assert pd.notna(vwap_features["vwap_upper_1.0std"].iloc[-1])
    assert vwap_features["vwap_upper_1.0std"].iloc[-1] >= latest_vwap
    assert vwap_features["vwap_lower_1.0std"].iloc[-1] <= latest_vwap


def test_compute_volume_zscore(sample_ohlcv_data):
    """
    Tests that the volume z-score correctly identifies spikes.
    """
    volume = sample_ohlcv_data["volume"]
    z_score = compute_volume_zscore(volume, window=5)
    
    # The last value (500) is significantly higher than the previous 4
    # Mean of last 5: (180+300+350+400+450+500)/6 = 346
    # Mean of last 5 except last: (180+300+350+400)/4 = 307
    # The last z-score should be positive and significant
    assert z_score.iloc[-1] > 1.0


def test_compute_volume_flow_features_integration(sample_ohlcv_data):
    """
    Tests the main feature computation function to ensure it returns a valid vector.
    """
    features = compute_volume_flow_features(
        sample_ohlcv_data,
        vwap_window=5,
        volume_z_window=5
    )
    
    assert isinstance(features, dict)
    assert "vwap" in features
    assert "volume_z_score" in features
    assert "close_vwap_deviation" in features
    
    # Ensure no NaN values are returned
    for key, value in features.items():
        assert pd.notna(value), f"Feature '{key}' returned NaN"

def test_vwap_empty_input():
    """Tests that VWAP handles empty input gracefully."""
    result = compute_vwap(pd.Series(dtype=float), pd.Series(dtype=float), window=5)
    assert result.empty

def test_zscore_empty_input():
    """Tests that Z-score handles empty input gracefully."""
    result = compute_volume_zscore(pd.Series(dtype=float), window=5)
    assert result.empty

def test_zscore_short_input(sample_ohlcv_data):
    """Tests that Z-score handles input shorter than the window."""
    volume = sample_ohlcv_data['volume'].head(3)
    result = compute_volume_zscore(volume, window=5)
    assert result.isnull().all()