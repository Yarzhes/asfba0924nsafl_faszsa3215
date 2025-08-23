"""
Core Mathematical Utilities for Technical Analysis

This module provides a collection of pure, vectorized functions for calculating
common technical indicators. These functions are designed to be high-performance,
relying on NumPy and Pandas for their computations.

Design Principles:
- Pure Functions: Each function takes data as input and returns a result
  without causing side effects. This makes them predictable and easy to test.
- Vectorized: Operations are applied to entire arrays (Pandas Series) at
  once, which is significantly faster than iterative calculations in Python.
- Dependency-Free Core Logic: Functions primarily use `pandas` and `numpy`,
  and can optionally use the `ta` library for more complex indicators if needed.
- Clear and Explicit: Function signatures are type-hinted and their purposes
  are clearly documented.
"""

import pandas as pd
from ta.momentum import rsi
from ta.volatility import AverageTrueRange


def rolling_ema(data: pd.Series, period: int) -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA) for a given data series.

    Args:
        data: A Pandas Series of prices (e.g., close prices).
        period: The time period for the EMA (e.g., 20, 50).

    Returns:
        A Pandas Series containing the EMA values.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input 'data' must be a Pandas Series.")
    if period <= 0:
        raise ValueError("EMA 'period' must be a positive integer.")

    return data.ewm(span=period, adjust=False).mean()


def rolling_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).

    This function is a wrapper around the `ta` library's RSI implementation.

    Args:
        close_prices: A Pandas Series of closing prices.
        period: The time period for the RSI calculation (default is 14).

    Returns:
        A Pandas Series containing the RSI values (ranging from 0 to 100).
    """
    if not isinstance(close_prices, pd.Series):
        raise TypeError("Input 'close_prices' must be a Pandas Series.")
    if period <= 0:
        raise ValueError("RSI 'period' must be a positive integer.")

    return rsi(close=close_prices, window=period)


def rolling_atr(
    high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int = 14
) -> pd.Series:
    """
    Calculates the Average True Range (ATR).

    This function is a wrapper around the `ta` library's ATR implementation.

    Args:
        high_prices: A Pandas Series of high prices.
        low_prices: A Pandas Series of low prices.
        close_prices: A Pandas Series of closing prices.
        period: The time period for the ATR calculation (default is 14).

    Returns:
        A Pandas Series containing the ATR values.
    """
    if not all(isinstance(s, pd.Series) for s in [high_prices, low_prices, close_prices]):
        raise TypeError("Inputs 'high_prices', 'low_prices', and 'close_prices' must be Pandas Series.")
    if period <= 0:
        raise ValueError("ATR 'period' must be a positive integer.")

    atr_indicator = AverageTrueRange(
        high=high_prices, low=low_prices, close=close_prices, window=period
    )
    return atr_indicator.average_true_range()