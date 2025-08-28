"""Extreme Event Protection - Legacy Circuit Breaker (Sprint 65)

Backwards-compatible simple circuit breaker with enhanced capabilities.
This maintains the original API while adding Sprint 65 features.
"""
from dataclasses import dataclass
from typing import Optional
from .shock_detector import ShockDetector, ShockConfig
from .circuit_breaker_engine import CircuitBreakerEngine, CircuitPolicy, CircuitLevel


@dataclass 
class CircuitState:
    """Legacy circuit state - maintained for backwards compatibility."""
    triggered: bool = False
    cooldown_bars_left: int = 0
    reason: Optional[str] = None


class CircuitBreaker:
    """Legacy circuit breaker with Sprint 65 enhancements.
    
    Maintains backwards compatibility while providing access to advanced features.
    For new code, use CircuitBreakerEngine directly.
    """
    
    def __init__(self, k_sigma: float = 6.0, cooldown_bars: int = 5, symbol: str = ""):
        self.k_sigma = float(k_sigma)
        self.cooldown_bars = int(cooldown_bars)
        self.symbol = symbol
        
        # Legacy state
        self.state = CircuitState()
        
        # Enhanced engine (optional, created on first advanced usage)
        self._enhanced_engine: Optional[CircuitBreakerEngine] = None
        self._shock_detector: Optional[ShockDetector] = None
    
    def _ensure_enhanced_engine(self) -> CircuitBreakerEngine:
        """Lazy initialization of enhanced engine."""
        if self._enhanced_engine is None:
            # Create shock detector with legacy-compatible config
            shock_config = ShockConfig(
                warn_k_sigma=max(2.0, self.k_sigma - 2),
                derisk_k_sigma=max(3.0, self.k_sigma - 1), 
                flatten_k_sigma=self.k_sigma,
                halt_k_sigma=self.k_sigma + 2
            )
            self._shock_detector = ShockDetector(shock_config, self.symbol)
            
            # Create circuit policy with legacy-compatible settings
            policy = CircuitPolicy(
                halt_threshold=4.0,
                flatten_threshold=3.0,
                derisk_threshold=2.0,
                warn_threshold=1.0,
                halt_cooldown_bars=self.cooldown_bars,
                flatten_cooldown_bars=max(self.cooldown_bars - 2, 3),
                derisk_cooldown_bars=max(self.cooldown_bars - 3, 2),
                warn_cooldown_bars=max(self.cooldown_bars - 4, 1)
            )
            
            self._enhanced_engine = CircuitBreakerEngine(self._shock_detector, policy, self.symbol)
        
        return self._enhanced_engine

    def check_and_trigger(self, ret_pct: float, sigma: float, 
                         spread_z: Optional[float] = None, 
                         vpin_toxic: bool = False,
                         **kwargs) -> CircuitState:
        """Legacy API with enhanced detection."""
        
        # Use enhanced engine if available data
        if any(kwargs.get(k) is not None for k in ['rv_z', 'depth_drop', 'lambda_z', 'venue_health']):
            engine = self._ensure_enhanced_engine()
            
            # Update shock detector with available data
            if sigma is not None and ret_pct is not None:
                timestamp_ms = int(__import__('time').time() * 1000)
                # Simulate price update to compute return
                current_price = 100.0  # Dummy price
                previous_price = current_price / (1 + ret_pct)
                self._shock_detector.update_price(timestamp_ms - 1000, previous_price)
                self._shock_detector.update_price(timestamp_ms, current_price)
            
            if kwargs.get('rv_z') is not None:
                rv = abs(kwargs['rv_z']) * sigma if sigma else abs(kwargs['rv_z'])
                self._shock_detector.update_realized_vol(timestamp_ms, rv)
            
            if spread_z is not None and kwargs.get('spread_bps'):
                self._shock_detector.update_orderbook(
                    timestamp_ms, kwargs['spread_bps'], 1000, 1000
                )
            
            # Run enhanced detection
            enhanced_state = engine.update(bar_close=True)
            
            # Map enhanced state to legacy state
            if enhanced_state.level in [CircuitLevel.FLATTEN, CircuitLevel.HALT]:
                self.state.triggered = True
                self.state.reason = enhanced_state.reason_codes[0] if enhanced_state.reason_codes else 'ENHANCED_TRIGGER'
                self.state.cooldown_bars_left = enhanced_state.cooldown_bars_left
            else:
                self.state.triggered = False
                self.state.reason = None
                self.state.cooldown_bars_left = 0
            
            return self.state
        
        # Original legacy logic
        if sigma is not None and ret_pct <= -abs(self.k_sigma) * abs(sigma):
            self.state.triggered = True
            self.state.cooldown_bars_left = self.cooldown_bars
            self.state.reason = 'RETURN_SPIKE'
            return self.state
        
        if vpin_toxic and (spread_z is None or spread_z > 1.0):
            self.state.triggered = True
            self.state.cooldown_bars_left = self.cooldown_bars
            self.state.reason = 'VPIN_TOXIC'
            return self.state
        
        return self.state

    def tick_bar(self) -> CircuitState:
        """Update cooldown timers."""
        if self.state.cooldown_bars_left > 0:
            self.state.cooldown_bars_left -= 1
            if self.state.cooldown_bars_left <= 0:
                self.state.triggered = False
                self.state.reason = None
        
        # Also update enhanced engine if active
        if self._enhanced_engine is not None:
            self._enhanced_engine.update(bar_close=True)
        
        return self.state
    
    # New Sprint 65 methods (optional enhanced API)
    def get_enhanced_state(self) -> Optional['CircuitState']:
        """Get enhanced circuit state with full Sprint 65 features."""
        if self._enhanced_engine is None:
            return None
        return self._enhanced_engine.state
    
    def get_size_multiplier(self) -> float:
        """Get current size multiplier from enhanced engine."""
        if self._enhanced_engine is None:
            return 1.0 if not self.state.triggered else 0.0
        return self._enhanced_engine.get_effective_size_mult()
    
    def can_enter_position(self) -> bool:
        """Check if position entry is allowed."""
        if self._enhanced_engine is not None:
            allowed, _ = self._enhanced_engine.can_enter_position()
            return allowed
        return not self.state.triggered
    
    def get_telemetry(self) -> dict:
        """Get comprehensive telemetry."""
        base_telemetry = {
            "legacy_triggered": self.state.triggered,
            "legacy_reason": self.state.reason,
            "legacy_cooldown": self.state.cooldown_bars_left,
            "enhanced_engine_active": self._enhanced_engine is not None
        }
        
        if self._enhanced_engine is not None:
            base_telemetry["enhanced"] = self._enhanced_engine.get_telemetry()
        
        return base_telemetry
