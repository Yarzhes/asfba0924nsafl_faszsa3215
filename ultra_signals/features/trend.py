"""
Trend-Following Feature Calculation

This module provides functions for calculating trend-based technical indicators,
such as Exponential Moving Averages (EMAs). These features are crucial for
determining the overall direction and strength of the market.

The functions are designed to operate on Pandas DataFrames and return a
dictionary of the calculated feature values.
"""

from functools import lru_cache
from typing import Dict

import pandas as pd

import pandas_ta as ta
import numpy as np

from ultra_signals.core.mathutils import rolling_ema


def compute_trend_features(ohlcv: pd.DataFrame, ema_short: int = 20, ema_medium: int = 50, ema_long: int = 200, adx_period: int = 14) -> Dict[str, float]:
    """
    Computes a set of trend-following indicators (EMAs, ADX).
    """
    close = ohlcv['close']

    # Use periods from the parameters dictionary
    ema_short_period = ema_short
    ema_medium_period = ema_medium
    ema_long_period = ema_long

    # Calculate EMAs
    ema_short = close.rolling(window=ema_short_period).mean()
    ema_medium = close.rolling(window=ema_medium_period).mean()
    ema_long = close.rolling(window=ema_long_period).mean()

    # Calculate ADX using pandas_ta
    adx_df = ta.adx(ohlcv['high'], ohlcv['low'], ohlcv['close'], length=adx_period)
    adx = adx_df[f"ADX_{adx_period}"] if adx_df is not None else pd.Series([np.nan])


    # Return the latest value of each calculated indicator
    return {
        "ema_short": ema_short.iloc[-1],
        "ema_medium": ema_medium.iloc[-1],
        "ema_long": ema_long.iloc[-1],
        "adx": adx.iloc[-1]
    }