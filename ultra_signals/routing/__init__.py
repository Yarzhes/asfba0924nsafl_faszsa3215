"""Smart multi-venue routing package.

This package provides a minimal, testable router, cost model and helpers
to start implementing the Sprint 55 design.
"""

from .types import *
from .aggregator import Aggregator, AggregatedBook
from .cost_model import estimate_all_in_cost, CostBreakdown
from .router import Router, RouterDecision
from .health import HealthMonitor
from .twap_adapter import TWAPExecutor
from .vwap_adapter import VWAPExecutor, StrategySelector
from .telemetry import TelemetryLogger

__all__ = [
    "Aggregator",
    "AggregatedBook",
    "estimate_all_in_cost",
    "CostBreakdown",
    "Router",
    "RouterDecision",
    "HealthMonitor",
    "TWAPExecutor",
    "VWAPExecutor",
    "StrategySelector",
    "TelemetryLogger",
]
