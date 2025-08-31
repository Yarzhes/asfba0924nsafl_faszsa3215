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
    from loguru import logger
    
    logger.debug(f"[TREND CALC DEBUG] Input data shape: {ohlcv.shape}, periods: short={ema_short}, med={ema_medium}, long={ema_long}")
    logger.debug(f"[TREND CALC DEBUG] Data range: {ohlcv.index[0] if len(ohlcv) > 0 else 'empty'} to {ohlcv.index[-1] if len(ohlcv) > 0 else 'empty'}")
    
    # Check if we have sufficient data for the longest EMA
    min_required = max(ema_short, ema_medium, ema_long)
    if len(ohlcv) < min_required:
        logger.warning(f"[TREND CALC DEBUG] Insufficient data: {len(ohlcv)} rows, need at least {min_required} for longest EMA")
        # Return NaN values to indicate insufficient data
        return {
            "ema_short": np.nan,
            "ema_medium": np.nan,
            "ema_long": np.nan,
            "adx": np.nan
        }
    
    close = ohlcv['close']
    logger.debug(f"[TREND CALC DEBUG] Close prices sample: {close.tail(3).tolist()}")

    # Use periods from the parameters dictionary
    ema_short_period = ema_short
    ema_medium_period = ema_medium
    ema_long_period = ema_long

    # Calculate EMAs using exponential weighted moving average
    ema_short = close.ewm(span=ema_short_period, adjust=False).mean()
    ema_medium = close.ewm(span=ema_medium_period, adjust=False).mean()
    ema_long = close.ewm(span=ema_long_period, adjust=False).mean()
    
    logger.debug(f"[TREND CALC DEBUG] EMA results: short={ema_short.iloc[-1]}, med={ema_medium.iloc[-1]}, long={ema_long.iloc[-1]}")

    # Calculate ADX using pandas_ta with error handling
    try:
        adx_df = ta.adx(ohlcv['high'], ohlcv['low'], ohlcv['close'], length=adx_period)
        if adx_df is not None and f"ADX_{adx_period}" in adx_df.columns:
            adx = adx_df[f"ADX_{adx_period}"]
        else:
            adx = pd.Series([np.nan] * len(ohlcv), index=ohlcv.index)
    except Exception:
        adx = pd.Series([np.nan] * len(ohlcv), index=ohlcv.index)


    # Return the latest value of each calculated indicator
    result = {
        "ema_short": ema_short.iloc[-1],
        "ema_medium": ema_medium.iloc[-1],
        "ema_long": ema_long.iloc[-1],
        "adx": adx.iloc[-1]
    }
    
    logger.debug(f"[TREND CALC DEBUG] Final result: {result}")
    return result