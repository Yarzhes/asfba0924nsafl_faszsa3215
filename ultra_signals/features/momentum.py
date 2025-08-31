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
    Compute momentum-based features from OHLCV data.
    
    Args:
        ohlcv: DataFrame with OHLC data
        rsi_period: Period for RSI calculation
        macd_fast: Fast EMA period for MACD
        macd_slow: Slow EMA period for MACD
        macd_signal: Signal line EMA period for MACD
        
    Returns:
        Dictionary containing momentum features
    """
    close = ohlcv['close']
    
    # Check if we have sufficient data - be more lenient with requirements
    min_periods_rsi = max(rsi_period + 1, 15)  # RSI needs at least rsi_period + 1
    min_periods_macd = max(macd_slow + macd_signal, 35)  # MACD needs slow + signal periods
    
    # Initialize default values
    rsi_value = 50.0  # Neutral RSI instead of NaN
    macd_line_value = 0.0  # Neutral MACD instead of NaN
    macd_signal_value = 0.0
    macd_hist_value = 0.0
    
    # Calculate RSI with more lenient data requirements
    if len(close) >= min_periods_rsi:
        try:
            rsi_series = rsi(close=close, window=rsi_period)
            if rsi_series is not None and not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]):
                rsi_value = float(rsi_series.iloc[-1])
        except Exception:
            pass  # Keep default neutral value

    # Calculate MACD components with more lenient data requirements
    if len(close) >= min_periods_macd:
        try:
            macd_indicator = MACD(
                close=close,
                window_slow=macd_slow,
                window_fast=macd_fast,
                window_sign=macd_signal
            )
            
            macd_line = macd_indicator.macd()
            macd_signal_line = macd_indicator.macd_signal()
            macd_hist = macd_indicator.macd_diff()
            
            # Use values if they're valid
            if macd_line is not None and not macd_line.empty and not pd.isna(macd_line.iloc[-1]):
                macd_line_value = float(macd_line.iloc[-1])
            if macd_signal_line is not None and not macd_signal_line.empty and not pd.isna(macd_signal_line.iloc[-1]):
                macd_signal_value = float(macd_signal_line.iloc[-1])
            if macd_hist is not None and not macd_hist.empty and not pd.isna(macd_hist.iloc[-1]):
                macd_hist_value = float(macd_hist.iloc[-1])
                
        except Exception:
            pass  # Keep default neutral values
    
    return {
        "rsi": rsi_value,
        "macd_line": macd_line_value,
        "macd_signal": macd_signal_value,
        "macd_hist": macd_hist_value,
    }