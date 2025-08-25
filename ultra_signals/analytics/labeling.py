"""Labeling & Target Engineering (Sprint 12)
=================================================
This module creates CLEAN training labels for supervised model development
or offline analytics. It focuses on SHORT-HORIZON intraday futures trading.

Goal (Sprint 12):
  - Produce direction labels based on 20–40 minute forward behavior
    (parameterised as `horizon_min_bars` .. `horizon_max_bars`).
  - Use VOLATILITY-SCALED thresholds so labels adapt to changing ATR.
  - Add an explicit NO-TRADE / CHOP label (0) when price action is
    ambiguous: both sides trigger OR neither side triggers the threshold.
  - Provide helper to merge labels with an OHLCV frame (no leakage).

Core Concept
------------
For each bar (index i) we look AHEAD between `horizon_min_bars` and
`horizon_max_bars` bars (inclusive) and measure:
  * MFE% (max favourable excursion): (max(high[i+min..max]) - close[i]) / close[i]
  * MAE% (max adverse excursion): (min(low[i+min..max]) - close[i]) / close[i]

We compute a volatility threshold:
    vol_threshold_pct = atr_mult * ATR[i] / close[i]

Label Rules
-----------
LONG  (+1):  MFE% >= vol_threshold_pct AND MAE% > -vol_threshold_pct
SHORT (-1):  MAE% <= -vol_threshold_pct AND MFE% <  vol_threshold_pct
NO_TRADE (0): all other cases (both triggered, or neither triggered)

No Leakage Guarantee
--------------------
Labels for bar i ONLY look at future highs/lows up to i + horizon_max_bars.
Any bars without a *complete* forward window receive label = NaN so that
you can safely drop them in a training set.

Typical Usage
-------------
    from ultra_signals.analytics.labeling import compute_vol_scaled_labels
    labels_df = compute_vol_scaled_labels(ohlcv_df, timeframe_minutes=5)
    # Merge with features on index (timestamp)

Inputs Expected
---------------
`ohlcv` must be a DataFrame indexed by timestamp with columns:
  ['open','high','low','close','volume'] (standard in this codebase).

Parameters
----------
- timeframe_minutes: ONLY used to express that a 4–8 bar horizon on a 5m
  series equals 20–40 minutes (purely informational; logic uses bar counts).
- atr_period: period for ATR (reused from mathutils / ta library).
- atr_mult: scaling factor for threshold band.

Returned DataFrame Columns
--------------------------
  label              : {+1, 0, -1} or NaN (if not enough future bars)
  vol_threshold_pct  : volatility threshold used for that bar
  mfe_pct            : forward max favourable excursion percentage
  mae_pct            : forward max adverse excursion percentage
  fwd_close_ret_pct  : close-to-close return at horizon_max_bars
  horizon_min_bars / horizon_max_bars : echo of params (metadata)

This keeps target engineering SEPARATE from live engine logic, avoiding
polluting real-time code with offline-only constructs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ultra_signals.core.mathutils import rolling_atr


@dataclass
class LabelSpec:
    """Configuration container (optional convenience)."""
    horizon_min_bars: int = 4
    horizon_max_bars: int = 8
    atr_period: int = 14
    atr_mult: float = 0.8


def _validate_inputs(ohlcv: pd.DataFrame, spec: LabelSpec) -> None:
    if ohlcv is None or ohlcv.empty:
        raise ValueError("OHLCV DataFrame is empty.")
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in ohlcv.columns:
            raise ValueError(f"Missing required OHLCV column: {col}")
    if spec.horizon_min_bars <= 0 or spec.horizon_max_bars <= 0:
        raise ValueError("Horizons must be positive integers.")
    if spec.horizon_min_bars > spec.horizon_max_bars:
        raise ValueError("horizon_min_bars cannot exceed horizon_max_bars.")


def compute_vol_scaled_labels(
    ohlcv: pd.DataFrame,
    timeframe_minutes: int = 5,
    horizon_min_bars: int = 4,
    horizon_max_bars: int = 8,
    atr_period: int = 14,
    atr_mult: float = 0.8,
) -> pd.DataFrame:
    """Compute volatility-scaled directional labels.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Indexed by timestamp. Must contain ['open','high','low','close','volume'].
    timeframe_minutes : int
        Informational only (e.g. 5m bars => 4–8 bars = 20–40 minutes).
    horizon_min_bars : int
        Earliest forward bar index (inclusive) to start evaluating.
    horizon_max_bars : int
        Furthest forward bar index (inclusive) for evaluation.
    atr_period : int
        ATR period for volatility scaling.
    atr_mult : float
        Multiplier applied to ATR/close to form threshold.

    Returns
    -------
    pd.DataFrame indexed like input with:
        label, vol_threshold_pct, mfe_pct, mae_pct, fwd_close_ret_pct,
        horizon_min_bars, horizon_max_bars
    """
    spec = LabelSpec(horizon_min_bars, horizon_max_bars, atr_period, atr_mult)
    _validate_inputs(ohlcv, spec)

    close = ohlcv["close"].astype(float)
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)

    # ATR (absolute) then convert to percent of price
    atr_series = rolling_atr(high, low, close, period=atr_period)
    vol_thr_pct = (atr_mult * atr_series / close).replace([np.inf, -np.inf], np.nan)

    n = len(ohlcv)
    labels = np.full(n, np.nan)
    mfe_pct = np.full(n, np.nan)
    mae_pct = np.full(n, np.nan)
    fwd_close_ret_pct = np.full(n, np.nan)

    max_h = horizon_max_bars
    min_h = horizon_min_bars

    # Convert to numpy arrays for slicing speed
    close_vals = close.values
    high_vals = high.values
    low_vals = low.values
    vol_thr_arr = vol_thr_pct.values

    for i in range(n):
        end = i + max_h
        start = i + min_h
        if end >= n:  # not enough future bars to form full window
            continue
        # Forward slices inclusive of horizon bounds
        f_highs = high_vals[start : end + 1]
        f_lows = low_vals[start : end + 1]
        cur_close = close_vals[i]
        if cur_close == 0 or np.isnan(cur_close):
            continue
        max_future_high = np.nanmax(f_highs)
        min_future_low = np.nanmin(f_lows)
        mfep = (max_future_high - cur_close) / cur_close
        maep = (min_future_low - cur_close) / cur_close
        mfe_pct[i] = mfep
        mae_pct[i] = maep
        fwd_close_ret_pct[i] = (close_vals[end] - cur_close) / cur_close

        thr = vol_thr_arr[i]
        if np.isnan(thr):
            continue

        long_cond = mfep >= thr and maep > -thr
        short_cond = maep <= -thr and mfep < thr
        if long_cond and not short_cond:
            labels[i] = 1
        elif short_cond and not long_cond:
            labels[i] = -1
        else:
            labels[i] = 0  # ambiguous or both triggered or neither triggered

    out = pd.DataFrame(
        {
            "label": labels,
            "vol_threshold_pct": vol_thr_pct.values,
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "fwd_close_ret_pct": fwd_close_ret_pct,
            "horizon_min_bars": spec.horizon_min_bars,
            "horizon_max_bars": spec.horizon_max_bars,
        },
        index=ohlcv.index,
    )
    return out


def merge_features_and_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join features with labels (common timestamp index) dropping NaN labels.

    This helper is purely for offline dataset preparation and is NOT used
    in the live engine to avoid any risk of future leakage.
    """
    merged = features_df.join(labels_df, how="left")
    return merged[~merged["label"].isna()].copy()


__all__ = [
    "LabelSpec",
    "compute_vol_scaled_labels",
    "merge_features_and_labels",
]
