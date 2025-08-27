"""Harmonic pattern detector (simplified ratio template matcher stub).

This stub scans recent pivots (basic zigzag approximation via percent move)
and attempts to classify a small subset of harmonic patterns based on XABCD
ratio templates. Real implementation will replace the pivot extraction and
ratio tolerance logic.
"""
from __future__ import annotations

from typing import List
import pandas as pd

from ultra_signals.core.custom_types import PatternInstance, PatternType, PatternDirection
from .base import BasePatternDetector


class HarmonicDetector(BasePatternDetector):
    @property
    def name(self) -> str:
        return "harmonic"

    def generate(self, symbol: str, timeframe: str, ohlcv: pd.DataFrame) -> List[PatternInstance]:
        # Placeholder: not performing real harmonic detection yet.
        return []

    def compute_targets_and_stops(self, instance: PatternInstance, ohlcv: pd.DataFrame) -> PatternInstance:
        # Generic harmonic projection: target1 0.618 of CD, target2 1.0 of CD extension
        if instance.pat_type in (PatternType.GARTLEY, PatternType.BAT, PatternType.BUTTERFLY, PatternType.CRAB,
                                 PatternType.CYPHER, PatternType.SHARK):
            if instance.breakout_px and instance.measured_move_px:
                mm = instance.measured_move_px
                if instance.direction == PatternDirection.LONG:
                    instance.target1_px = instance.breakout_px + 0.618 * mm
                    instance.target2_px = instance.breakout_px + 1.0 * mm
                    if not instance.struct_stop_px and instance.neckline_px:
                        instance.struct_stop_px = instance.neckline_px * 0.985
                else:
                    instance.target1_px = instance.breakout_px - 0.618 * mm
                    instance.target2_px = instance.breakout_px - 1.0 * mm
                    if not instance.struct_stop_px and instance.neckline_px:
                        instance.struct_stop_px = instance.neckline_px * 1.015
        return instance
