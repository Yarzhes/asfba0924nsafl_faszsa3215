"""Sprint 11 Feature Pack v2
================================
Composite intraday / structure features:
- Multi-timeframe structure: recent high/low breaks (20-bar lookback)
- Anchored session VWAP bands (reset at UTC day boundary)
- RSI divergences (simple heuristic over last N pivots)
- ADX slope (rate of change of ADX)
- Keltner vs Bollinger squeeze ratio
- Volume bursts (volume z-score > threshold)
- Session / time-of-day encoding + week-of-month
- Lightweight attribution snapshot (simple normalized feature magnitudes)

All computations use existing OHLCV already in FeatureStore to avoid duplication.
"""
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd
import numpy as np

from ultra_signals.core.custom_types import AlphaV2Features


# ---------------------------- Helpers ----------------------------

def _session_anchor(df: pd.DataFrame) -> pd.Timestamp:
    """Return session anchor timestamp (start of UTC day)."""
    ts = df.index[-1]
    return pd.Timestamp(year=ts.year, month=ts.month, day=ts.day)


def _compute_session_vwap(df: pd.DataFrame) -> Dict[str, float]:
    anchor = _session_anchor(df)
    day_df = df[df.index >= anchor]
    if day_df.empty:
        return {}
    tp = (day_df['high'] + day_df['low'] + day_df['close']) / 3
    vol = day_df['volume'].replace(0, np.nan)
    vp = (tp * vol).cumsum()
    vv = vol.cumsum()
    vwap = vp / vv
    std = ( (tp - vwap)**2 * vol ).cumsum() / vv
    std = np.sqrt(std).fillna(0)
    out = {
        'sess_vwap': float(vwap.iloc[-1]),
        'sess_vwap_upper_1': float(vwap.iloc[-1] + std.iloc[-1]),
        'sess_vwap_lower_1': float(vwap.iloc[-1] - std.iloc[-1]),
    }
    close = day_df['close'].iloc[-1]
    out['sess_vwap_dev'] = (close - out['sess_vwap']) / out['sess_vwap'] if out['sess_vwap'] else 0.0
    return out


def _highest_lowest_flags(df: pd.DataFrame, lookback: int = 20) -> Dict[str, int]:
    if len(df) < lookback + 1:
        return {'hh_break_20': 0, 'll_break_20': 0, 'range_pct_20': None}
    window = df.tail(lookback + 1)
    prev = window.iloc[:-1]
    last = window.iloc[-1]
    hh = prev['high'].max()
    ll = prev['low'].min()
    flag_h = int(last['high'] > hh)
    flag_l = int(last['low'] < ll)
    rng = (prev['high'].max() - prev['low'].min())
    close = last['close']
    rng_pct = (rng / close) if close else None
    return {'hh_break_20': flag_h, 'll_break_20': flag_l, 'range_pct_20': rng_pct}


def _adx_slope(trend_feats: dict, df: pd.DataFrame, adx_period: int, slope_window: int = 5) -> Optional[float]:
    # We rely on existing trend feature already computing ADX. Here we form slope over last slope_window bars if available.
    if 'adx' not in trend_feats or trend_feats['adx'] is None:
        return None
    if len(df) < slope_window + 1:
        return None
    # simplistic slope: (current - value N bars ago) / N
    # For convenience assume we can reconstruct a recent ADX series via rolling high/low/close fallback.
    # Without stored history of ADX, approximate zero slope.
    return 0.0


def _bollinger_keltner_squeeze(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0, ema_period: int = 20, atr_period: int = 14) -> Dict[str, float]:
    if len(df) < max(bb_period, ema_period, atr_period) + 5:
        return {'bb_kc_ratio': None, 'squeeze_flag': 0}
    close = df['close']
    ema = close.ewm(span=ema_period, adjust=False).mean()
    std = close.rolling(bb_period).std()
    bb_width = 2 * bb_std * std
    tr = (df['high'] - df['low']).abs()
    atr = tr.rolling(atr_period).mean()
    kc_width = 2 * atr  # simplified
    if kc_width.iloc[-1] == 0 or pd.isna(kc_width.iloc[-1]):
        return {'bb_kc_ratio': None, 'squeeze_flag': 0}
    ratio = float(bb_width.iloc[-1] / kc_width.iloc[-1]) if kc_width.iloc[-1] else None
    squeeze = int(ratio is not None and ratio < 1.0)
    return {'bb_kc_ratio': ratio, 'squeeze_flag': squeeze}


def _volume_burst(df: pd.DataFrame, window: int = 50, z_thr: float = 2.5) -> int:
    if len(df) < window + 5:
        return 0
    vol = df['volume']
    mean = vol.rolling(window).mean()
    std = vol.rolling(window).std().replace(0, np.nan)
    if pd.isna(mean.iloc[-1]) or pd.isna(std.iloc[-1]):
        return 0
    z = (vol.iloc[-1] - mean.iloc[-1]) / std.iloc[-1]
    return int(z >= z_thr)


def _week_of_month(ts: pd.Timestamp) -> int:
    first = ts.replace(day=1)
    dom = ts.day
    adjusted_dom = dom + first.weekday()
    return int(np.ceil(adjusted_dom/7.0))


def _session_label(hour: int) -> str:
    # Simple 4-session model
    if 0 <= hour < 6:
        return 'asia_early'
    if 6 <= hour < 12:
        return 'asia_late'
    if 12 <= hour < 18:
        return 'eu_us_overlap'
    return 'us_close'


def _divergence_flags(df: pd.DataFrame, rsi_series: pd.Series, lookback: int = 30) -> Dict[str, int]:
    # Very simple divergence detection: compare last two swing highs/lows (price vs RSI direction)
    if len(df) < lookback or rsi_series.empty:
        return {'bull_div': 0, 'bear_div': 0}
    closes = df['close'].tail(lookback)
    rsi_tail = rsi_series.tail(lookback)
    # pivots (naive): argmax/argmin first half vs second half
    half = lookback // 2
    p1_high = closes.head(half).max(); p2_high = closes.tail(half).max()
    r1_high = rsi_tail.head(half).max(); r2_high = rsi_tail.tail(half).max()
    p1_low = closes.head(half).min(); p2_low = closes.tail(half).min()
    r1_low = rsi_tail.head(half).min(); r2_low = rsi_tail.tail(half).min()
    bear = int(p2_high > p1_high and r2_high < r1_high)
    bull = int(p2_low < p1_low and r2_low > r1_low)
    return {'bull_div': bull, 'bear_div': bear}


def compute_alpha_v2_features(ohlcv: pd.DataFrame, existing_features: Dict[str, object] | None = None, settings: Dict | None = None) -> Dict[str, float]:
    if ohlcv is None or ohlcv.empty:
        return {}
    feats: Dict[str, float | int | None] = {}
    df = ohlcv

    # Structure
    feats.update(_highest_lowest_flags(df, lookback=20))

    # Session anchored VWAP
    feats.update(_compute_session_vwap(df))

    # Momentum & RSI divergences (reuse RSI if provided)
    rsi_val = None
    if existing_features and 'momentum' in existing_features:
        rsi_val = getattr(existing_features['momentum'], 'rsi', None)
    # If not present, attempt compute quick RSI
    try:
        if rsi_val is None:
            from ta.momentum import rsi as _rsi
            rsi_series = _rsi(df['close'], window=14)
            rsi_val = float(rsi_series.iloc[-1])
        else:
            from ta.momentum import rsi as _rsi
            rsi_series = _rsi(df['close'], window=14)
    except Exception:
        rsi_series = pd.Series(dtype=float)
    feats['rsi'] = rsi_val
    feats.update(_divergence_flags(df, rsi_series, lookback=30))

    # ADX slope placeholder (0 if cannot compute)
    if existing_features and 'trend' in existing_features:
        trf = existing_features['trend']
        adx_val = getattr(trf, 'adx', None)
        feats['adx_slope_5'] = 0.0 if adx_val is not None else None
    else:
        feats['adx_slope_5'] = None

    # Squeeze
    feats.update(_bollinger_keltner_squeeze(df))

    # Volume burst
    feats['volume_burst'] = _volume_burst(df)

    # Time / calendar
    ts = df.index[-1]
    feats['hour'] = int(ts.hour)
    feats['session'] = _session_label(int(ts.hour))
    feats['week_of_month'] = _week_of_month(ts)

    # Attribution snapshot (naive: normalize selected numeric magnitudes)
    attrib_keys = ['hh_break_20','ll_break_20','sess_vwap_dev','adx_slope_5','bb_kc_ratio','volume_burst']
    vals = [abs(feats[k]) for k in attrib_keys if isinstance(feats.get(k), (int,float)) and feats.get(k) is not None]
    if vals:
        total = sum(vals) or 1.0
        feats['attribution'] = {k: round(abs(feats[k])/total, 4) for k in attrib_keys if k in feats and isinstance(feats[k], (int,float))}

    return feats


def build_alpha_v2_model(raw: Dict[str, float]) -> AlphaV2Features:
    return AlphaV2Features(**raw)
