"""
Tests for the v1 feature computation functions (trend, momentum, volatility).
"""

import numpy as np
import pandas as pd
import pytest

from ultra_signals.features import compute_trend_features as compute_trend
from ultra_signals.features import compute_momentum_features as compute_momentum
from ultra_signals.features import compute_volatility_features as compute_volatility

# A smaller, more predictable dataframe for specific checks
@pytest.fixture
def simple_ohlcv_data() -> pd.DataFrame:
    data = {
        'open': [100, 101, 102, 103, 104, 105],
        'high': [101, 102, 103, 104, 105, 106],
        'low': [99, 100, 101, 102, 103, 104],
        'close': [101, 102, 103, 104, 105, 106],
        'volume': [10, 10, 10, 10, 10, 10],
    }
    return pd.DataFrame(data)

def test_compute_trend(ohlcv_fixture, settings_fixture):
    """
    Tests the `compute_trend` function for correctness.
    """
    params = settings_fixture.features.trend.model_dump()
    features = compute_trend(ohlcv_fixture, **params)

    assert isinstance(features, dict)
    
    # Check that all expected keys are present
    expected_keys = [
        "ema_short",
        "ema_medium",
        "ema_long"
    ]
    assert all(key in features for key in expected_keys)
    
    # Check that values are valid floats
    assert all(isinstance(value, float) for value in features.values())
    assert not any(np.isnan(value) for value in features.values())


def test_compute_momentum(ohlcv_fixture, settings_fixture):
    """
    Tests the `compute_momentum` function.
    """
    params = settings_fixture.features.momentum.model_dump()
    features = compute_momentum(ohlcv_fixture, **params)

    assert isinstance(features, dict)
    
    expected_keys = ["rsi", "macd_line", "macd_signal", "macd_hist"]
    assert all(key in features for key in expected_keys)
    
    assert isinstance(features["rsi"], float)
    assert 0 <= features["rsi"] <= 100
    
    assert not any(np.isnan(value) for value in features.values())


def test_compute_volatility(ohlcv_fixture, settings_fixture):
    """
    Tests the `compute_volatility` function.
    """
    params = settings_fixture.features.volatility.model_dump()
    features = compute_volatility(ohlcv_fixture, **params)

    assert isinstance(features, dict)
    
    expected_keys = ["atr", "bbands_upper", "bbands_lower"]
    assert all(key in features for key in expected_keys)
    
    # ATR must be positive
    assert features["atr"] > 0
    
    # Bollinger bands should bracket the middle band
    assert features["bbands_upper"] >= features["bbands_lower"]

    assert not any(np.isnan(value) for value in features.values())

def test_insufficient_data_handling(simple_ohlcv_data, settings_fixture):
    """
    Tests that feature computations handle insufficient data gracefully.
    The underlying `ta` library will return NaNs, this test confirms that behavior.
    """
    # Use a dataframe that is shorter than the required periods
    short_df = simple_ohlcv_data.head(5)
    
    # Test with a long EMA period
    trend_params = {"ema_short": 2, "ema_medium": 3, "ema_long": 10} # ema_long is > len(df)
    trend_features = compute_trend(short_df, **trend_params)
    # With adjust=False, EMA can produce a value even with fewer points than the window
    assert isinstance(trend_features['ema_short'], float)

    # Test with default RSI period (14) which is > len(df)
    momentum_params = settings_fixture.features.momentum.model_dump()
    momentum_features = compute_momentum(short_df, **momentum_params)
    assert np.isnan(momentum_features['rsi'])