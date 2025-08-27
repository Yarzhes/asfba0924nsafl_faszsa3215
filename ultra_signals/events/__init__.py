"""Events subsystem (Sprint 28).

Provides:
 - Provider interfaces for external economic / crypto incident calendars.
 - Local SQLite backed event store.
 - Classification + gating logic to veto / dampen trades around high-impact events.

Initial lightweight implementation – providers are stubs returning empty lists
until real data sources are wired. Gating + store + config schema are functional
so backtest / live engine can already integrate observe-only mode.
"""

from .gating import evaluate as evaluate_gate, GateDecision, stats as gate_stats, reset_caches  # re-export
from . import store, classifier, providers

__all__ = [
    'evaluate_gate','GateDecision','gate_stats','reset_caches','store','classifier','providers'
]
