"""Execution layer package (Sprint 24).

Modules:
  pricing  : Smart maker-first limit pricing with fences & taker fallback.
  brackets : Bracket/OCO order construction (SL/TP ladder, BE + trailing).
  algos    : Child-order execution algos (TWAP, Iceberg, POV prototype).
  guards   : Spread/latency/slippage/flip guards used pre/post send.
  metrics  : Execution attribution & aggregation helpers.

These are intentionally lightweight adapters; business logic stays pure so both
live and backtest paths can share them.
"""
from .pricing import ExecPlan, build_exec_plan
from .brackets import build_brackets
from .algos import plan_child_orders
from .guards import pre_trade_guards
from .metrics import ExecAttribution

__all__ = [
    'ExecPlan', 'build_exec_plan', 'build_brackets', 'plan_child_orders', 'pre_trade_guards', 'ExecAttribution'
]
