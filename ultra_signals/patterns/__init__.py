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
