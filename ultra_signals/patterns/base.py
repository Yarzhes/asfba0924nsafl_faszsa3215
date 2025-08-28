from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any
import pandas as pd


class BarType(Enum):
    TIME = "TIME"
    HEIKIN_ASHI = "HEIKIN_ASHI"
    RENKO = "RENKO"
    RANGE = "RANGE"
    KAGI = "KAGI"
    POINT_FIGURE = "POINT_FIGURE"


@dataclass(frozen=True)
class Bar:
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class Bars:
    bars: List[Bar] = field(default_factory=list)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Bars":
        rows = []
        for idx, r in df.iterrows():
            rows.append(
                Bar(
                    pd.Timestamp(idx),
                    float(r.get('open', float('nan'))),
                    float(r.get('high', float('nan'))),
                    float(r.get('low', float('nan'))),
                    float(r.get('close', float('nan'))),
                    float(r.get('volume', 0.0)),
                )
            )
        return cls(rows)


@dataclass
class DetectedPattern:
    name: str
    direction: str  # BULL | BEAR | NEUTRAL
    strength: float  # 0..1
    start_idx: int
    end_idx: int
    meta: Dict[str, Any] = field(default_factory=dict)


class BarAdapter:
    def transform(self, bars: Bars, **params) -> Bars:
        raise NotImplementedError()


class PatternDetector:
    def detect(self, bars: Bars) -> List[DetectedPattern]:
        raise NotImplementedError()


from abc import ABC, abstractmethod
from ultra_signals.core.custom_types import (
    PatternInstance,
    PatternStage,
    PatternType as CT_PatternType,
    PatternDirection as CT_PatternDirection,
)


class BasePatternDetector(ABC):
    """Abstract base used by the package's classical/harmonic/engine detectors.

    Minimal required API implemented here so existing detectors can subclass
    without duplication.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate(self, symbol: str, timeframe: str, ohlcv: pd.DataFrame) -> List[PatternInstance]:
        """Produce zero or more PatternInstance objects for the given OHLCV window."""

    def compute_targets_and_stops(self, instance: PatternInstance, ohlcv: pd.DataFrame) -> PatternInstance:
        return instance

    def _new_instance(self, ts_ms: int, symbol: str, timeframe: str, pat_type: CT_PatternType,
                      direction: CT_PatternDirection, reason_codes: List[str] | None = None) -> PatternInstance:
        inst = PatternInstance(
            ts=ts_ms,
            symbol=symbol,
            timeframe=timeframe,
            pat_type=pat_type,
            direction=direction,
            stage=PatternStage.FORMING,
        )
        inst.reason_codes = reason_codes or []
        inst.confluence = []
        return inst


