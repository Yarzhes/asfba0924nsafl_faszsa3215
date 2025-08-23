"""
Momentum-Based Feature Calculation

This module provides functions for calculating momentum indicators like RSI and MACD.
These features help in identifying overbought or oversold conditions and potential
reversals in the market.
"""

from functools import lru_cache
from typing import Dict

import pandas as pd
from ta.momentum import rsi
from ta.trend import MACD

from ultra_signals.core.mathutils import rolling_rsi


def compute_momentum_features(ohlcv: pd.DataFrame, rsi_period: int = 14, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> Dict[str, float]:
    """
    Computes a set of momentum indicators (RSI, MACD).
    """
    close = ohlcv['close']

    macd_fast_period = macd_fast
    macd_slow_period = macd_slow
    macd_signal_period = macd_signal

    # Calculate RSI
    rsi_series = rsi(close=close, window=rsi_period)

    # Calculate MACD components
    macd_indicator = MACD(
        close=close,
        window_slow=macd_slow_period,
        window_fast=macd_fast_period,
        window_sign=macd_signal_period
    )
    
    macd_line = macd_indicator.macd()
    macd_signal_line = macd_indicator.macd_signal()
    macd_hist = macd_indicator.macd_diff()
    
    return {
        "rsi": rsi_series.iloc[-1],
        "macd_line": macd_line.iloc[-1],
        "macd_signal": macd_signal_line.iloc[-1],
        "macd_hist": macd_hist.iloc[-1],
    }