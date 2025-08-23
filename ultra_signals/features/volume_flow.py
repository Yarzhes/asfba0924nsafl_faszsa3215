"""
Volume Flow and VWAP Features
-----------------------------

This module provides functions for calculating features related to trading volume
and Volume-Weighted Average Price (VWAP). These indicators are crucial for
understanding market conviction, identifying accumulation/distribution, and
gauging price levels of high institutional interest.

- VWAP (Volume-Weighted Average Price): Provides the average price a security
  has traded at throughout the day, based on both volume and price. It's often
  used as a benchmark by institutional traders. We calculate it on a rolling
  basis over a specified period.

- VWAP Bands: Standard deviation bands placed above and below the VWAP, which
  can signal overbought or oversold conditions relative to the volume-weighted
  average.

- Volume Z-Score: Measures how far the current bar's volume is from the rolling
  average volume in terms of standard deviations. It helps to quickly identify
  unusual volume spikes that may signal significant market events.
"""

from typing import Dict, Deque
from collections import deque
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from ultra_signals.core.custom_types import FeatureVector
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultra_signals.core.feature_store import FeatureStore


def compute_vwap(
    typical_price: pd.Series,
    volume: pd.Series,
    window: int = 20,
    std_devs: tuple[float, ...] = (1.0, 2.0),
) -> pd.DataFrame:
    """
    Calculates rolling VWAP and its standard deviation bands.

    Args:
        typical_price: Series of typical price ((High + Low + Close) / 3).
        volume: Series of volume data.
        window: The rolling window period for VWAP calculation.
        std_devs: A tuple of standard deviation multipliers for the bands.

    Returns:
        A DataFrame with columns for VWAP, and upper/lower bands for each
        standard deviation specified.
    """
    if typical_price.empty or volume.empty:
        return pd.DataFrame()

    vp = typical_price * volume
    rolling_vp = vp.rolling(window=window).sum()
    rolling_vol = volume.rolling(window=window).sum()

    vwap = rolling_vp / rolling_vol

    # Calculate rolling standard deviation of price from VWAP
    price_deviation_sq = ((typical_price - vwap) ** 2 * volume).rolling(window=window).sum()
    # Fill NaNs with 0 to ensure bands are always present
    vwap_std = np.sqrt(price_deviation_sq / rolling_vol).fillna(0)

    features = pd.DataFrame(index=typical_price.index)
    features["vwap"] = vwap

    for std in std_devs:
        features[f"vwap_upper_{std}std"] = vwap + (vwap_std * std)
        features[f"vwap_lower_{std}std"] = vwap - (vwap_std * std)

    return features


def compute_volume_zscore(volume: pd.Series, window: int = 50) -> pd.Series:
    """
    Calculates the Z-score of volume over a rolling window.

    Args:
        volume: Series of volume data.
        window: The rolling window period.

    Returns:
        A Series containing the volume z-score.
    """
    if volume.empty or len(volume) < window:
        return pd.Series(index=volume.index, dtype=float)

    rolling_mean = volume.rolling(window=window).mean()
    rolling_std = volume.rolling(window=window).std()

    # Avoid division by zero for periods of no volume
    rolling_std = rolling_std.replace(0, np.nan)
    
    z_score = (volume - rolling_mean) / rolling_std
    return z_score.ffill() # Forward-fill NaNs at the start


def compute_volume_flow_features(
    ohlcv: pd.DataFrame,
    vwap_window: int = 20,
    vwap_std_devs: tuple[float, ...] = (1.0, 2.0),
    volume_z_window: int = 50,
    **kwargs,
) -> Dict[str, float]:
    """
    Computes a vector of all volume, flow, and VWAP related features.
    """
    if ohlcv.empty:
        return {}

    features = {}

    # --- VWAP Calculation ---
    typical_price = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3
    vwap_features = compute_vwap(
        typical_price, ohlcv["volume"], vwap_window, vwap_std_devs
    )
    if not vwap_features.empty and 'vwap' in vwap_features.columns:
        vwap_val = vwap_features['vwap'].iloc[-1]
        if pd.notna(vwap_val):
            features['vwap'] = vwap_val
            close_price = ohlcv["close"].iloc[-1]
            features["close_vwap_deviation"] = (close_price - vwap_val) / vwap_val

    # --- Volume Z-Score ---
    volume_z = compute_volume_zscore(ohlcv["volume"], volume_z_window)
    if not volume_z.empty:
        features["volume_z_score"] = volume_z.iloc[-1]

    return {k: (v if pd.notna(v) else 0.0) for k, v in features.items()}
