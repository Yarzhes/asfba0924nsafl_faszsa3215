"""Lightweight ASCII overlay renderer for pattern instances.

Produces a simple textual depiction of price range and pattern key levels
for logging/Telegram previews. Future: PNG rendering with matplotlib.
"""
from __future__ import annotations

from typing import List
import pandas as pd
from ultra_signals.core.custom_types import PatternInstance


def ascii_overlay(ohlcv: pd.DataFrame, patterns: List[PatternInstance], width: int = 60, bars: int = 60) -> str:
    if ohlcv is None or ohlcv.empty:
        return "<no data>"
    df = ohlcv.tail(bars)
    highs = df['high'].astype(float)
    lows = df['low'].astype(float)
    closes = df['close'].astype(float)
    hi = float(highs.max())
    lo = float(lows.min())
    span = hi - lo or 1.0
    # map value -> row index (0 top)
    rows = []
    levels = []
    for p in patterns[:5]:
        for lvl_name in ['neckline_px','breakout_px','target1_px','target2_px','struct_stop_px']:
            v = getattr(p, lvl_name, None)
            if v is not None:
                levels.append((lvl_name, v, p.pat_type.value[:6]))
    levels_dedup = {}
    for name, v, tag in levels:
        key = (name, round(v, 2))
        if key not in levels_dedup:
            levels_dedup[key] = (name, v, tag)
    level_rows = {}
    for name, v, tag in levels_dedup.values():
        r = int((1 - (v - lo) / span) * 10)
        level_rows.setdefault(r, []).append(f"{name}:{tag}")
    # build 11 rows
    for r in range(11):
        y_val = hi - r * span / 10
        line = [" "] * width
        # mark price path (sample closes to width positions)
        sample = closes.iloc[max(0, len(closes)-width):].values
        for i, c in enumerate(sample):
            rr = int((1 - (c - lo) / span) * 10)
            if rr == r:
                line[i] = "·"
        # add levels
        if r in level_rows:
            line[-15:] = list("|" + ",".join(level_rows[r])[:14])
        rows.append(f"{y_val:>10.2f} |" + "".join(line))
    return "\n".join(rows)
