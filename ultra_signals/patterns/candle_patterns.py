"""Classical candlestick pattern detectors (lightweight, pure-Python).
"""
from typing import List, Dict
from .base import Bars, DetectedPattern
import math


def _body_size(b):
    return abs(b.close - b.open)


def _upper_wick(b):
    return max(0.0, b.high - max(b.close, b.open))


def _lower_wick(b):
    return max(0.0, min(b.close, b.open) - b.low)


def doji(bars: Bars) -> List[DetectedPattern]:
    out: List[DetectedPattern] = []
    if not bars.bars:
        return out
    b = bars.bars[-1]
    body = _body_size(b)
    rng = b.high - b.low if b.high > b.low else 1.0
    if body / rng < 0.1:
        out.append(DetectedPattern('doji', 'NEUTRAL', 0.6, len(bars.bars)-1, len(bars.bars)-1, {}))
    return out


def hammer(bars: Bars) -> List[DetectedPattern]:
    out: List[DetectedPattern] = []
    if not bars.bars:
        return out
    b = bars.bars[-1]
    body = _body_size(b)
    lower = _lower_wick(b)
    upper = _upper_wick(b)
    # allow upper wick to be <= body (some implementations treat small upper wick as acceptable)
    if lower > 2 * body and upper <= body:
        out.append(DetectedPattern('hammer', 'BULL', min(1.0, lower / (lower + body)), len(bars.bars)-1, len(bars.bars)-1, {}))
    return out


def shooting_star(bars: Bars) -> List[DetectedPattern]:
    out: List[DetectedPattern] = []
    if not bars.bars:
        return out
    b = bars.bars[-1]
    body = _body_size(b)
    upper = _upper_wick(b)
    lower = _lower_wick(b)
    if upper > 2 * body and lower < body:
        out.append(DetectedPattern('shooting_star', 'BEAR', min(1.0, upper / (upper + body)), len(bars.bars)-1, len(bars.bars)-1, {}))
    return out


def engulfing(bars: Bars) -> List[DetectedPattern]:
    out: List[DetectedPattern] = []
    if len(bars.bars) < 2:
        return out
    p = bars.bars[-2]
    c = bars.bars[-1]
    # bullish engulfing
    if c.close > c.open and p.close < p.open and c.open < p.close and c.close > p.open:
        strength = min(1.0, abs((c.close - c.open) / max(1e-8, p.close - p.open)))
        out.append(DetectedPattern('bullish_engulfing', 'BULL', strength, len(bars.bars)-2, len(bars.bars)-1, {}))
    # bearish engulfing
    if c.close < c.open and p.close > p.open and c.open > p.close and c.close < p.open:
        strength = min(1.0, abs((c.open - c.close) / max(1e-8, p.open - p.close)))
        out.append(DetectedPattern('bearish_engulfing', 'BEAR', strength, len(bars.bars)-2, len(bars.bars)-1, {}))
    return out


class CandlestickPatternLibrary:
    def __init__(self):
        self.single = [doji, hammer, shooting_star]
        self.dual = [engulfing]

    def detect(self, bars: Bars):
        detected: List[DetectedPattern] = []
        for f in self.single:
            try:
                detected.extend(f(bars))
            except Exception:
                pass
        for f in self.dual:
            try:
                detected.extend(f(bars))
            except Exception:
                pass
        # basic merging: if multiple conflict, pick top strength per name
        by_name = {}
        for d in detected:
            old = by_name.get(d.name)
            if old is None or d.strength > old.strength:
                by_name[d.name] = d
        return list(by_name.values())
