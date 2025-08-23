"""
Tests for the core mathematical utility functions.
"""

import numpy as np
import pandas as pd
import pytest

from ultra_signals.core.mathutils import rolling_ema, rolling_rsi, rolling_atr


def test_rolling_ema(ohlcv_fixture):
    """
    Tests the rolling_ema function for correctness.
    """
    close_prices = ohlcv_fixture['close']
    ema_10 = rolling_ema(close_prices, period=10)

    # EMA should be a Series of the same length
    assert isinstance(ema_10, pd.Series)
    assert len(ema_10) == len(close_prices)

    # The last value should be a valid number
    assert not np.isnan(ema_10.iloc[-1])

    # Compare with pandas' own implementation (they should be identical)
    pandas_ema = close_prices.ewm(span=10, adjust=False).mean()
    pd.testing.assert_series_equal(ema_10, pandas_ema)


def test_rolling_rsi(ohlcv_fixture):
    """
    Tests the rolling_rsi function.
    """
    close_prices = ohlcv_fixture['close']
    rsi_14 = rolling_rsi(close_prices, period=14)

    assert isinstance(rsi_14, pd.Series)
    assert len(rsi_14) == len(close_prices)
    
    # RSI values must be between 0 and 100
    assert rsi_14.min() >= 0
    assert rsi_14.max() <= 100
    
    # The first few values will be NaN, which is expected
    assert np.isnan(rsi_14.iloc[0])
    assert not np.isnan(rsi_14.iloc[-1])


def test_rolling_atr(ohlcv_fixture):
    """
    Tests the rolling_atr function.
    """
    high, low, close = ohlcv_fixture['high'], ohlcv_fixture['low'], ohlcv_fixture['close']
    atr_14 = rolling_atr(high, low, close, period=14)

    assert isinstance(atr_14, pd.Series)
    assert len(atr_14) == len(close)

    # ATR must be non-negative
    assert atr_14.min() >= 0
    
    # The last value should be a valid number
    assert not np.isnan(atr_14.iloc[-1])


def test_invalid_input_types():
    """
    Tests that functions raise TypeErrors for incorrect input types.
    """
    # Using a list instead of a Series
    invalid_data = [1, 2, 3, 4, 5]
    
    with pytest.raises(TypeError):
        rolling_ema(invalid_data, period=3)

    with pytest.raises(TypeError):
        rolling_rsi(invalid_data, period=3)
        
    with pytest.raises(TypeError):
        # Pass one valid series and two invalid lists
        valid_series = pd.Series(invalid_data)
        rolling_atr(valid_series, invalid_data, invalid_data, period=3)


def test_invalid_period_values(ohlcv_fixture):
    """
    Tests that functions raise ValueErrors for invalid period arguments.
    """
    data = ohlcv_fixture['close']
    
    with pytest.raises(ValueError, match="must be a positive integer"):
        rolling_ema(data, period=0)
        
    with pytest.raises(ValueError, match="must be a positive integer"):
        rolling_ema(data, period=-5)
        
    with pytest.raises(ValueError, match="must be a positive integer"):
        rolling_rsi(data, period=0)
        
    with pytest.raises(ValueError, match="must be a positive integer"):
        rolling_atr(data, data, data, period=0)