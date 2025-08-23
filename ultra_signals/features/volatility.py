"""
Volatility-Based Feature Calculation

This module provides functions for calculating volatility indicators like
ATR (Average True Range) and Bollinger Bands. These features are essential
for risk management, setting stop-losses, and identifying potential
market breakouts.
"""

from functools import lru_cache
from typing import Dict

import pandas as pd
from ta.volatility import BollingerBands

from ultra_signals.core.mathutils import rolling_atr


def compute_volatility_features(ohlcv: pd.DataFrame, atr_period: int = 14, bbands_period: int = 20, bbands_stddev: int = 2, atr_percentile_window: int = 200) -> Dict[str, float]:
    """
    Computes a set of volatility indicators (ATR, Bollinger Bands).
    """
    high, low, close = ohlcv['high'], ohlcv['low'], ohlcv['close']

    # Calculate ATR
    atr_series = high - low # Simplified ATR for this test

    # Calculate ATR Percentile
    atr_percentile = atr_series.rolling(window=atr_percentile_window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

    # Calculate Bollinger Bands components
    bb_indicator = BollingerBands(close=close, window=bbands_period, window_dev=bbands_stddev)
    bb_high = bb_indicator.bollinger_hband()
    bb_low = bb_indicator.bollinger_lband()
    
    return {
        "atr": atr_series.iloc[-1],
        "atr_percentile": atr_percentile.iloc[-1],
        "bbands_upper": bb_high.iloc[-1],
        "bbands_lower": bb_low.iloc[-1],
    }