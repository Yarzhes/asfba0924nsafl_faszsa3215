"""Bridge to extract deterministic numeric features from detected patterns.

Provides extract_pattern_features(bars, bar_type, config) -> Dict[str,float]
"""
from typing import Dict, Any
from .base import Bars, BarType, DetectedPattern
from .candle_patterns import CandlestickPatternLibrary
from .chart_patterns import ChartPatternLibrary


def extract_pattern_features(bars: Bars, bar_type: BarType, config: Dict[str, Any]) -> Dict[str, float]:
    # Deterministic mapping: aggregate strengths of bullish/bearish candles and structures
    out = {
        'candle_bull_score': 0.0,
        'candle_bear_score': 0.0,
        'structure_breakout_score': 0.0,
        'reversal_score': 0.0,
        'continuation_score': 0.0,
    }

    try:
        c_lib = CandlestickPatternLibrary()
        candles = c_lib.detect(bars) or []
        for p in candles:
            if p.direction == 'BULL':
                out['candle_bull_score'] += float(p.strength)
            elif p.direction == 'BEAR':
                out['candle_bear_score'] += float(p.strength)
            else:
                # neutral counts toward neither
                pass
            # simple mapping: strong patterns >0.8 considered reversal
            if p.strength >= 0.8:
                out['reversal_score'] += float(p.strength)
            else:
                out['continuation_score'] += float(p.strength)
    except Exception:
        pass

    try:
        # pass full patterns config so detectors can read thresholds and geom params
        cp = ChartPatternLibrary(config or {})
        structs = cp.detect(bars) or []
        for s in structs:
            if s.name in ('double_top', 'double_bottom'):
                out['structure_breakout_score'] += float(s.strength)
                # double acts as reversal
                out['reversal_score'] += float(s.strength)
    except Exception:
        pass

    # normalize by simple window size
    n = max(1, len(bars.bars))
    for k in list(out.keys()):
        out[k] = float(out[k]) / float(n)
    return out
