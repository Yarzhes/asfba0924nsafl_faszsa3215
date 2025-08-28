"""Liquidity / risk helpers (L-VaR, execution hints, circuit breakers).

Lightweight package exposing core risk primitives used by the signal engine and sizing.

Sprint 65 adds comprehensive extreme event protection:
- ShockDetector: Multi-signal shock detection
- CircuitBreakerEngine: Tiered circuit breakers with hysteresis  
- ExecutionSafetyAdapter: Safe position flattening
- CircuitBreakerAlerts: Telegram notifications
- ExtremeEventProtectionManager: Unified integration interface
"""
from .lvar import LVarEngine
from .exec_adapter import ExecAdapter, SizeSuggestion
from .circuit_breaker import CircuitBreaker

# Sprint 65 - Extreme Event Protection
from .shock_detector import ShockDetector, ShockConfig, ShockFeatures, ShockTrigger
from .circuit_breaker_engine import (
    CircuitBreakerEngine, 
    CircuitPolicy, 
    CircuitState as EnhancedCircuitState,
    CircuitLevel, 
    CircuitAction
)
from .execution_safety import (
    ExecutionSafetyAdapter, 
    SafeExitConfig, 
    ExitStyle, 
    ExitOrder, 
    FlattenerState
)
from .circuit_alerts import CircuitBreakerAlerts, AlertConfig
from .extreme_event_protection import (
    ExtremeEventProtectionManager,
    ExtremeEventStatus, 
    create_extreme_event_protection
)

__all__ = [
    # Legacy components
    "LVarEngine", 
    "ExecAdapter", 
    "SizeSuggestion", 
    "CircuitBreaker",
    
    # Sprint 65 - Extreme Event Protection
    "ShockDetector",
    "ShockConfig", 
    "ShockFeatures",
    "ShockTrigger",
    "CircuitBreakerEngine",
    "CircuitPolicy",
    "EnhancedCircuitState",
    "CircuitLevel",
    "CircuitAction", 
    "ExecutionSafetyAdapter",
    "SafeExitConfig",
    "ExitStyle",
    "ExitOrder",
    "FlattenerState",
    "CircuitBreakerAlerts",
    "AlertConfig",
    "ExtremeEventProtectionManager",
    "ExtremeEventStatus",
    "create_extreme_event_protection"
]
