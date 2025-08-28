"""Pattern engine package export.

Provides a simple PatternEngine facade expected by FeatureStore.
"""
from .base import Bar, Bars, BarType, DetectedPattern, BarAdapter, PatternDetector
from .feature_bridge import extract_pattern_features
from .bar_adapters import HeikinAshiAdapter, RenkoAdapter, get_adapter

from typing import List, Dict, Any


class PatternEngine:
    """Lightweight orchestrator that runs detectors and adapters.

    For now it exposes a minimal API used by FeatureStore:
      - with_default_detectors(cfg) -> PatternEngine
      - on_bar(symbol, timeframe, ohlcv_df) -> List[Dict] (PatternInstance-like)
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config or {}
        # detectors could be added here in future

    @classmethod
    def with_default_detectors(cls, cfg: Dict[str, Any]):
        return cls(cfg)

    def on_bar(self, symbol: str, timeframe: str, ohlcv_df) -> List[Dict]:
        # Convert ohlcv_df (pandas DataFrame) to Bars dataclass for adapters/detectors
        try:
            bars = Bars.from_dataframe(ohlcv_df)
            # choose adapter based on cfg
            bt = (self.cfg.get('bar_types') or ['TIME'])[0]
        except Exception:
            return []

        # create a simple pattern snapshot surface that matches existing PatternInstance expectations
        # For backward compatibility we keep it as list of dicts
        feat = extract_pattern_features(bars, BarType.TIME, self.cfg)
        # Create dummy PatternInstance-like dicts
        out = []
        # expose aggregated buckets and top patterns for storage
        out.append({'name': 'pattern_summary', 'meta': feat})
        return out
"""Sprint 44 Pattern Recognition Engine package.

Provides a hybrid (rules + ML scoring) modular architecture for classical
patterns, harmonics, volume profile structures, fractal state and S/R level
interactions. Detectors produce PatternInstance objects (see custom_types).

High-level pipeline (executed per new bar per timeframe):
 1. Candidate generation (geometry / pivot extraction)
 2. Rule validation & attribute enrichment
 3. ML scoring + probability calibration (optional, pluggable)
 4. Lifecycle state machine update (forming -> confirmed -> failed)
 5. Conflict resolution / de-duplication
 6. Target/stop projection & confidence tagging

Minimal scaffolding here; concrete detectors will be fleshed out incrementally.
"""

from .engine import PatternEngine
from .base import BasePatternDetector

__all__ = ["PatternEngine", "BasePatternDetector"]
