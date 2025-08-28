"""Bar adapters: Heikin-Ashi, Renko, and stubs for others.
"""
from .base import Bars, Bar, BarType
from typing import List, Optional
import pandas as pd
import numpy as np


class HeikinAshiAdapter:
    """Transforms time-series bars into Heikin-Ashi bars."""

    def transform(self, bars: Bars, **params) -> Bars:
        if not bars.bars:
            return Bars([])
        out = []
        prev_ha_open = None
        for i, b in enumerate(bars.bars):
            ha_close = (b.open + b.high + b.low + b.close) / 4.0
            if i == 0:
                ha_open = (b.open + b.close) / 2.0
            else:
                ha_open = (prev_ha_open + prev_ha_close) / 2.0
            ha_high = max(b.high, ha_open, ha_close)
            ha_low = min(b.low, ha_open, ha_close)
            out.append(Bar(b.ts, ha_open, ha_high, ha_low, ha_close, b.volume))
            prev_ha_open = ha_open
            prev_ha_close = ha_close
        return Bars(out)


class RenkoAdapter:
    """Simple Renko brick builder. box_size can be numeric or 'auto_atr'.

    For performance we implement an O(n) brick builder using prices.
    If box_size == 'auto_atr', estimate via ATR of closes with period atr_period.
    """

    def transform(self, bars: Bars, box_size: Optional[float] = None, atr_period: int = 14, **params) -> Bars:
        if not bars.bars:
            return Bars([])
        closes = np.array([b.close for b in bars.bars], dtype=float)
        if box_size is None or box_size == "auto_atr":
            # approximated ATR via simple True Range over closes (not ideal but fast)
            highs = np.array([b.high for b in bars.bars], dtype=float)
            lows = np.array([b.low for b in bars.bars], dtype=float)
            tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1)))[1:]
            if len(tr) < 1:
                box = max(1e-8, closes[-1] * 0.001)
            else:
                atr_est = float(np.mean(tr[-min(len(tr), atr_period):]))
                box = max(1e-8, atr_est)
        else:
            box = float(box_size)

        bricks: List[Bar] = []
        # Start from first close
        last_brick_price = closes[0]
        last_ts = bars.bars[0].ts
        for i, p in enumerate(closes[1:], start=1):
            diff = p - last_brick_price
            steps = int(np.floor(abs(diff) / box))
            if steps >= 1:
                direction = 1 if diff > 0 else -1
                for s in range(steps):
                    brick_price = last_brick_price + direction * box
                    # create brick with open=last_brick_price, close=brick_price, high/low accordingly
                    if direction > 0:
                        b = Bar(bars.bars[i].ts, last_brick_price, brick_price, last_brick_price, brick_price, 0.0)
                    else:
                        b = Bar(bars.bars[i].ts, last_brick_price, last_brick_price, brick_price, brick_price, 0.0)
                    bricks.append(b)
                    last_brick_price = brick_price
                    last_ts = bars.bars[i].ts

        return Bars(bricks)


def RangeAdapter(*args, **kwargs):
    """TODO: implement Range bars adapter. Placeholder raises NotImplementedError."""
    raise NotImplementedError("RangeAdapter is a TODO: design and implement range bars adapter")


def KagiAdapter(*args, **kwargs):
    """TODO: implement Kagi adapter. Placeholder."""
    raise NotImplementedError("KagiAdapter is a TODO: design and implement Kagi adapter")


def PointFigureAdapter(*args, **kwargs):
    """TODO: implement Point & Figure adapter. Placeholder."""
    raise NotImplementedError("PointFigureAdapter is a TODO: design and implement P&F adapter")


def get_adapter(bar_type: BarType):
    if bar_type == BarType.HEIKIN_ASHI:
        return HeikinAshiAdapter()
    if bar_type == BarType.RENKO:
        return RenkoAdapter()
    if bar_type == BarType.TIME:
        # identity adapter
        class _Id:
            def transform(self, bars, **params):
                return bars
        return _Id()
    # stubs
    if bar_type == BarType.RANGE:
        return RangeAdapter
    if bar_type == BarType.KAGI:
        return KagiAdapter
    if bar_type == BarType.POINT_FIGURE:
        return PointFigureAdapter
    return None
