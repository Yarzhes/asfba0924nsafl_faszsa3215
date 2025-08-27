"""High Timeframe (HTF) Feature Cache & Accessors (Sprint 30 Multi-Timeframe Confirmation)

Provides light wrappers around the existing FeatureStore so the Multi-Timeframe
Confirmation gate can fetch higher timeframe (HTF) indicator values cheaply.

The cache does NOT recompute indicators. It reuses the already-computed feature
objects (trend, momentum, volatility, volume_flow) and derives a few helpers:
  * macd_slope  (difference of current vs previous macd_line)
  * price_above_vwap
  * staleness flag (based on settings['mtc']['staleness_secs'])

Returned dataclass fields intentionally mirror what the MTC gate consumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
from loguru import logger

try:  # type hints only
    from ultra_signals.core.feature_store import FeatureStore
except Exception:  # pragma: no cover
    FeatureStore = object  # type: ignore


@dataclass(slots=True)
class HTFFeatures:
    timeframe: str
    ts: int  # epoch seconds of matched bar
    ema21: Optional[float]
    ema200: Optional[float]
    adx: Optional[float]
    macd_line: Optional[float]
    macd_signal: Optional[float]
    macd_hist: Optional[float]
    macd_slope: Optional[float]
    rsi: Optional[float]
    atr_percentile: Optional[float]
    vwap: Optional[float]
    price: Optional[float]
    price_above_vwap: Optional[bool]
    stale: bool = False

    def as_dict(self) -> Dict[str, Any]:  # convenience
        return {
            "tf": self.timeframe,
            "ts": self.ts,
            "ema21": self.ema21,
            "ema200": self.ema200,
            "adx": self.adx,
            "macd_line": self.macd_line,
            "macd_signal": self.macd_signal,
            "macd_hist": self.macd_hist,
            "macd_slope": self.macd_slope,
            "rsi": self.rsi,
            "atr_percentile": self.atr_percentile,
            "vwap": self.vwap,
            "price": self.price,
            "price_above_vwap": self.price_above_vwap,
            "stale": self.stale,
        }


class HTFFeatureCache:
    # Intentionally avoid strict type annotations referencing internal classes
    def __init__(self, feature_store, settings):
        self._fs = feature_store
        self._settings = settings or {}
        self._staleness_secs = ((self._settings.get("mtc") or {}).get("staleness_secs") or {})

    def get_htf_features(self, symbol: str, timeframe: str, ts_epoch: int) -> Optional[HTFFeatures]:
        try:
            ts = pd.to_datetime(int(ts_epoch), unit="s")
        except Exception:
            return None
        feats = self._fs.get_features(symbol, timeframe, ts, nearest=True)
        if not feats:
            return None
        # Locate actual matched bar timestamp
        bar_ts = None
        try:
            series = getattr(self._fs, "_feature_cache", {}).get(symbol, {}).get(timeframe, {})
            keys = [k for k in series.keys() if k <= ts]
            if keys:
                bar_ts = max(keys)
        except Exception:
            bar_ts = None
        if bar_ts is None:
            bar_ts = ts
        bar_epoch = int(pd.Timestamp(bar_ts).timestamp())

        trend = feats.get("trend") if isinstance(feats, dict) else None
        momentum = feats.get("momentum") if isinstance(feats, dict) else None
        volatility = feats.get("volatility") if isinstance(feats, dict) else None
        vol_flow = feats.get("volume_flow") if isinstance(feats, dict) else None

        ema21 = getattr(trend, "ema_short", None) or getattr(trend, "ema21", None)
        ema200 = getattr(trend, "ema_long", None) or getattr(trend, "ema200", None)
        adx = getattr(trend, "adx", None)
        macd_line = getattr(momentum, "macd_line", None)
        macd_signal = getattr(momentum, "macd_signal", None)
        macd_hist = getattr(momentum, "macd_hist", None)
        rsi = getattr(momentum, "rsi", None)
        atr_percentile = None
        if volatility is not None:
            for name in ("atr_percentile", "atr_pct", "atrp"):
                v = getattr(volatility, name, None)
                if v is not None:
                    atr_percentile = v
                    break
        vwap = getattr(vol_flow, "vwap", None)
        price = None
        try:
            df = self._fs.get_ohlcv(symbol, timeframe)
            if df is not None and not df.empty:
                if bar_ts in df.index:
                    price = float(df.loc[bar_ts, "close"])
                else:
                    price = float(df["close"].iloc[-1])
        except Exception:
            price = None
        price_above_vwap = None
        if price is not None and vwap is not None:
            try:
                price_above_vwap = price >= vwap
            except Exception:
                price_above_vwap = None

        # MACD slope
        macd_slope = None
        try:
            series = getattr(self._fs, "_feature_cache", {}).get(symbol, {}).get(timeframe, {})
            ordered = sorted([k for k in series.keys() if k <= bar_ts])
            if len(ordered) >= 2 and macd_line is not None:
                prev_feats = series[ordered[-2]]
                prev_mom = prev_feats.get("momentum") if isinstance(prev_feats, dict) else None
                prev_macd = getattr(prev_mom, "macd_line", None)
                if prev_macd is not None:
                    macd_slope = float(macd_line) - float(prev_macd)
        except Exception:
            macd_slope = None

        # Staleness
        stale = False
        try:
            max_age = int(self._staleness_secs.get(timeframe.lower(), 0) or self._staleness_secs.get(timeframe, 0) or 0)
            if max_age > 0:
                if (ts_epoch - bar_epoch) > max_age:
                    stale = True
        except Exception:
            stale = False

        return HTFFeatures(
            timeframe=timeframe,
            ts=bar_epoch,
            ema21=_to_float(ema21),
            ema200=_to_float(ema200),
            adx=_to_float(adx),
            macd_line=_to_float(macd_line),
            macd_signal=_to_float(macd_signal),
            macd_hist=_to_float(macd_hist),
            macd_slope=_to_float(macd_slope),
            rsi=_to_float(rsi),
            atr_percentile=_to_float(atr_percentile),
            vwap=_to_float(vwap),
            price=_to_float(price),
            price_above_vwap=price_above_vwap,
            stale=stale,
        )


def _to_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


__all__ = ["HTFFeatureCache", "HTFFeatures"]
