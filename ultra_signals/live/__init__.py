"""Live trading (Sprint 21) package scaffolding.

This folder contains the bounded async pipeline components introduced in
Sprint 21. All modules are designed to be additive and optional – importing
them will not affect existing backtest / realtime runner behaviour until the
new live CLI (`ultra_signals.apps.live_runner`) is invoked.
"""

__all__ = [
    "LiveRunner",
]
