"""Base interfaces for pattern detectors (Sprint 44).

Each detector focuses on a family of structures (e.g., classical chart
patterns, harmonics, volume profile anomalies). They ingest rolling OHLCV
windows and emit zero or more PatternInstance objects or updates.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any

import pandas as pd

from ultra_signals.core.custom_types import PatternInstance, PatternType, PatternStage, PatternDirection


class BasePatternDetector(ABC):
    """Abstract base class for all pattern detectors.

    Implementations should be stateless across symbols/timeframes where
    feasible; persistent life-cycle tracking is coordinated by PatternEngine.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @property
    @abstractmethod
    def name(self) -> str:  # short unique id
        raise NotImplementedError

    @property
    def types_emitted(self) -> List[PatternType]:  # override if restricted subset
        return []

    @abstractmethod
    def generate(self, symbol: str, timeframe: str, ohlcv: pd.DataFrame) -> List[PatternInstance]:
        """Return fresh pattern candidates (stage=FORMING) given latest window.

        Should not mutate existing instances; only produce new ones. Engine
        merges them with tracked state via stable hash_id or geometry digest.
        """
        raise NotImplementedError

    def refine(self, instance: PatternInstance, ohlcv: pd.DataFrame) -> PatternInstance:
        """Optional per-bar refinement callback for ongoing formations.

        Default: passthrough. Detectors can override to update neckline,
        targets, quality, etc. Must return the (possibly modified) instance.
        """
        return instance

    def compute_targets_and_stops(self, instance: PatternInstance, ohlcv: pd.DataFrame) -> PatternInstance:
        """Optional measured move / target & structural stop generation.

        Default is passthrough; detectors with specific geometry (e.g. head &
        shoulders, double top) can override. Engine will call after integrate
        if fields still missing.
        """
        return instance

    # --- Helper: build base instance (shared config policy) ---
    def _new_instance(
        self,
        *,
        ts_ms: int,
        symbol: str,
        timeframe: str,
        pat_type: PatternType,
        direction: PatternDirection,
        reason_codes: list[str] | None = None,
    ) -> PatternInstance:
        return PatternInstance(
            ts=ts_ms,
            symbol=symbol,
            timeframe=timeframe,
            pat_type=pat_type,
            direction=direction,
            stage=PatternStage.FORMING,
            reason_codes=reason_codes or [self.name],
            age_bars=0,
            freshness_bars=0,
        )
