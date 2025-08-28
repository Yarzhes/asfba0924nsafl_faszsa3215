"""Extreme Event Protection - Integration Module (Sprint 65)

Provides a unified interface for integrating extreme event protection
into the live trading system and other components.
"""
from __future__ import annotations
import time
import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from loguru import logger

from .shock_detector import ShockDetector, ShockConfig, ShockFeatures
from .circuit_breaker_engine import CircuitBreakerEngine, CircuitPolicy, CircuitLevel, CircuitState
from .execution_safety import ExecutionSafetyAdapter, SafeExitConfig
from .circuit_alerts import CircuitBreakerAlerts, AlertConfig
from ..core.config import ExtremeEventProtectionSettings


@dataclass
class ExtremeEventStatus:
    """Current status of extreme event protection system."""
    enabled: bool = True
    circuit_level: str = "normal"
    size_mult_current: float = 1.0
    leverage_cap_current: Optional[float] = None
    
    # Metrics
    threat_score: float = 0.0
    trigger_count: int = 0
    reason_codes: List[str] = None
    
    # Recovery info
    countdown_bars: int = 0
    recovery_stage: int = 0
    
    # Actions
    block_new_entries: bool = False
    cancel_resting_orders: bool = False
    flatten_positions: bool = False
    
    # Timestamps
    last_update_ms: int = 0
    triggered_at_ms: int = 0


class ExtremeEventProtectionManager:
    """Main integration point for extreme event protection.
    
    This class provides a clean interface for integrating Sprint 65
    extreme event protection into the live trading system.
    """
    
    def __init__(self, 
                 settings: ExtremeEventProtectionSettings,
                 symbol: str = "",
                 telegram_settings: Optional[Dict] = None):
        self.settings = settings
        self.symbol = symbol
        self.telegram_settings = telegram_settings or {}
        
        # Core components
        self.shock_detector: Optional[ShockDetector] = None
        self.circuit_engine: Optional[CircuitBreakerEngine] = None
        self.execution_adapter: Optional[ExecutionSafetyAdapter] = None
        self.alerts: Optional[CircuitBreakerAlerts] = None
        
        # State
        self.enabled = settings.enabled
        self.last_update_ms = 0
        self.metrics_cache: Dict[str, Any] = {}
        
        # Callbacks for integration
        self.callbacks: Dict[str, List[Callable]] = {
            "level_change": [],
            "size_mult_change": [],
            "cancel_orders": [],
            "flatten_positions": [],
            "alert_sent": []
        }
        
        # Initialize if enabled
        if self.enabled:
            self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all protection components."""
        try:
            # Shock detector
            shock_config = ShockConfig(
                return_windows_sec=self.settings.shock_detection.return_windows_sec,
                warn_k_sigma=self.settings.shock_detection.warn_k_sigma,
                derisk_k_sigma=self.settings.shock_detection.derisk_k_sigma,
                flatten_k_sigma=self.settings.shock_detection.flatten_k_sigma,
                halt_k_sigma=self.settings.shock_detection.halt_k_sigma,
                rv_warn_z=self.settings.shock_detection.rv_warn_z,
                rv_derisk_z=self.settings.shock_detection.rv_derisk_z,
                rv_flatten_z=self.settings.shock_detection.rv_flatten_z,
                spread_warn_z=self.settings.shock_detection.spread_warn_z,
                spread_derisk_z=self.settings.shock_detection.spread_derisk_z,
                depth_warn_drop_pct=self.settings.shock_detection.depth_warn_drop_pct,
                depth_derisk_drop_pct=self.settings.shock_detection.depth_derisk_drop_pct,
                vpin_warn_pctl=self.settings.shock_detection.vpin_warn_pctl,
                vpin_derisk_pctl=self.settings.shock_detection.vpin_derisk_pctl,
                vpin_flatten_pctl=self.settings.shock_detection.vpin_flatten_pctl,
                lambda_warn_z=self.settings.shock_detection.lambda_warn_z,
                lambda_derisk_z=self.settings.shock_detection.lambda_derisk_z,
                min_triggers_warn=self.settings.shock_detection.min_triggers_warn,
                min_triggers_derisk=self.settings.shock_detection.min_triggers_derisk,
                min_triggers_flatten=self.settings.shock_detection.min_triggers_flatten,
                min_triggers_halt=self.settings.shock_detection.min_triggers_halt
            )
            self.shock_detector = ShockDetector(shock_config, self.symbol)
            
            # Circuit breaker policy
            circuit_policy = CircuitPolicy(
                warn_threshold=self.settings.circuit_policy.warn_threshold,
                derisk_threshold=self.settings.circuit_policy.derisk_threshold,
                flatten_threshold=self.settings.circuit_policy.flatten_threshold,
                halt_threshold=self.settings.circuit_policy.halt_threshold,
                warn_exit_threshold=self.settings.circuit_policy.warn_exit_threshold,
                derisk_exit_threshold=self.settings.circuit_policy.derisk_exit_threshold,
                flatten_exit_threshold=self.settings.circuit_policy.flatten_exit_threshold,
                halt_exit_threshold=self.settings.circuit_policy.halt_exit_threshold,
                warn_cooldown_bars=self.settings.circuit_policy.warn_cooldown_bars,
                derisk_cooldown_bars=self.settings.circuit_policy.derisk_cooldown_bars,
                flatten_cooldown_bars=self.settings.circuit_policy.flatten_cooldown_bars,
                halt_cooldown_bars=self.settings.circuit_policy.halt_cooldown_bars,
                enable_staged_recovery=self.settings.circuit_policy.enable_staged_recovery,
                recovery_stages=self.settings.circuit_policy.recovery_stages,
                recovery_stage_bars=self.settings.circuit_policy.recovery_stage_bars
            )
            
            # Update policy actions with custom settings
            circuit_policy.actions[CircuitLevel.WARN].size_mult = self.settings.circuit_policy.warn_size_mult
            circuit_policy.actions[CircuitLevel.WARN].leverage_cap = self.settings.circuit_policy.warn_leverage_cap
            circuit_policy.actions[CircuitLevel.DERISK].size_mult = self.settings.circuit_policy.derisk_size_mult
            circuit_policy.actions[CircuitLevel.DERISK].leverage_cap = self.settings.circuit_policy.derisk_leverage_cap
            circuit_policy.actions[CircuitLevel.FLATTEN].size_mult = self.settings.circuit_policy.flatten_size_mult
            circuit_policy.actions[CircuitLevel.FLATTEN].leverage_cap = self.settings.circuit_policy.flatten_leverage_cap
            circuit_policy.actions[CircuitLevel.HALT].size_mult = self.settings.circuit_policy.halt_size_mult
            
            self.circuit_engine = CircuitBreakerEngine(self.shock_detector, circuit_policy, self.symbol)
            
            # Execution safety adapter
            safe_exit_config = SafeExitConfig(
                max_participation_rate=self.settings.safe_exit.max_participation_rate,
                slice_duration_sec=self.settings.safe_exit.slice_duration_sec,
                max_slices=self.settings.safe_exit.max_slices,
                passive_timeout_sec=self.settings.safe_exit.passive_timeout_sec,
                market_urgency_threshold=self.settings.safe_exit.market_urgency_threshold,
                allow_cross_spread=self.settings.safe_exit.allow_cross_spread,
                min_order_value_usd=self.settings.safe_exit.min_order_value_usd,
                venue_health_threshold=self.settings.safe_exit.venue_health_threshold
            )
            self.execution_adapter = ExecutionSafetyAdapter(safe_exit_config)
            
            # Alert system
            alert_config = AlertConfig(
                enabled=self.settings.alerts.enabled,
                rate_limit_sec=self.settings.alerts.rate_limit_sec,
                max_triggers_shown=self.settings.alerts.max_triggers_shown,
                include_countdown=self.settings.alerts.include_countdown,
                include_technical_details=self.settings.alerts.include_technical_details
            )
            self.alerts = CircuitBreakerAlerts(alert_config, self.symbol)
            
            # Register callbacks
            self._register_internal_callbacks()
            
            logger.info(f"[ExtremeEventProtection] Initialized for {self.symbol}")
            
        except Exception as e:
            logger.error(f"[ExtremeEventProtection] Initialization failed: {e}")
            self.enabled = False
    
    def _register_internal_callbacks(self) -> None:
        """Register internal callbacks between components."""
        if not self.circuit_engine:
            return
        
        # Register execution actions
        self.circuit_engine.register_callback("cancel_orders", self._handle_cancel_orders)
        self.circuit_engine.register_callback("flatten_positions", self._handle_flatten_positions)
        self.circuit_engine.register_callback("size_change", self._handle_size_change)
        self.circuit_engine.register_callback("alert", self._handle_alert)
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register external callback for events.
        
        Args:
            event_type: One of 'level_change', 'size_mult_change', 'cancel_orders', 
                       'flatten_positions', 'alert_sent'
            callback: Function to call on event
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            logger.warning(f"[ExtremeEventProtection] Unknown callback type: {event_type}")
    
    def inject_execution_dependencies(self, 
                                    order_manager=None,
                                    position_manager=None,
                                    venue_router=None,
                                    market_data=None) -> None:
        """Inject execution dependencies into safety adapter."""
        if self.execution_adapter:
            self.execution_adapter.inject_dependencies(
                order_manager=order_manager,
                position_manager=position_manager,
                venue_router=venue_router,
                market_data=market_data
            )
    
    def update_market_data(self, 
                          timestamp_ms: Optional[int] = None,
                          price: Optional[float] = None,
                          realized_vol: Optional[float] = None,
                          spread_bps: Optional[float] = None,
                          top_bid_qty: Optional[float] = None,
                          top_ask_qty: Optional[float] = None,
                          ref_depth: Optional[float] = None,
                          vpin: Optional[float] = None,
                          vpin_pctl: Optional[float] = None,
                          lambda_val: Optional[float] = None,
                          oi_change_pct: Optional[float] = None,
                          funding_rate_bps: Optional[float] = None,
                          venue_health: float = 1.0,
                          stablecoin_depeg_bps: Optional[float] = None) -> None:
        """Update market data for shock detection."""
        if not self.enabled or not self.shock_detector:
            return
        
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        
        # Update shock detector
        if price is not None:
            self.shock_detector.update_price(timestamp_ms, price)
        
        if realized_vol is not None:
            self.shock_detector.update_realized_vol(timestamp_ms, realized_vol)
        
        if spread_bps is not None and top_bid_qty is not None and top_ask_qty is not None:
            self.shock_detector.update_orderbook(
                timestamp_ms, spread_bps, top_bid_qty, top_ask_qty, ref_depth
            )
        
        if vpin is not None and vpin_pctl is not None:
            self.shock_detector.update_vpin(vpin, vpin_pctl)
        
        if lambda_val is not None:
            self.shock_detector.update_lambda(lambda_val)
        
        if oi_change_pct is not None or funding_rate_bps is not None:
            self.shock_detector.update_derivatives(oi_change_pct, funding_rate_bps)
    
    def update(self, bar_close: bool = False) -> ExtremeEventStatus:
        """Main update method - call this regularly (e.g., every bar close)."""
        if not self.enabled or not self.circuit_engine:
            return ExtremeEventStatus(enabled=False)
        
        try:
            # Update circuit breaker engine
            old_level = self.circuit_engine.state.level
            circuit_state = self.circuit_engine.update(bar_close=bar_close)
            
            # Check for level changes
            if circuit_state.level != old_level:
                self._notify_callbacks("level_change", {
                    "old_level": old_level.value,
                    "new_level": circuit_state.level.value,
                    "triggers": circuit_state.triggers,
                    "symbol": self.symbol
                })
            
            self.last_update_ms = int(time.time() * 1000)
            
            # Build status response
            status = self._build_status(circuit_state)
            
            # Cache metrics
            self.metrics_cache = self.get_telemetry()
            
            return status
            
        except Exception as e:
            logger.error(f"[ExtremeEventProtection] Update failed: {e}")
            return ExtremeEventStatus(enabled=False)
    
    def _build_status(self, circuit_state: CircuitState) -> ExtremeEventStatus:
        """Build status object from circuit state."""
        countdown_info = self.circuit_engine.get_resume_countdown()
        
        action = self.circuit_engine.policy.actions.get(circuit_state.level)
        
        return ExtremeEventStatus(
            enabled=self.enabled,
            circuit_level=circuit_state.level.value,
            size_mult_current=circuit_state.size_mult_current,
            leverage_cap_current=circuit_state.leverage_cap_current,
            threat_score=self._get_latest_threat_score(),
            trigger_count=len(circuit_state.triggers),
            reason_codes=circuit_state.reason_codes.copy() if circuit_state.reason_codes else [],
            countdown_bars=countdown_info.get("countdown_bars", 0),
            recovery_stage=countdown_info.get("recovery", {}).get("recovery_stage", 0),
            block_new_entries=action.block_new_entries if action else False,
            cancel_resting_orders=action.cancel_resting if action else False,
            flatten_positions=action.flatten_positions if action else False,
            last_update_ms=self.last_update_ms,
            triggered_at_ms=circuit_state.triggered_at_ms
        )
    
    def _get_latest_threat_score(self) -> float:
        """Get latest threat score from circuit engine."""
        if not self.circuit_engine or not self.circuit_engine.metrics_history:
            return 0.0
        return self.circuit_engine.metrics_history[-1][1]  # threat_score
    
    def can_enter_position(self, side: str = "long") -> tuple[bool, str]:
        """Check if new position entry is allowed."""
        if not self.enabled or not self.circuit_engine:
            return True, "protection_disabled"
        
        return self.circuit_engine.can_enter_position(side)
    
    def can_modify_position(self, is_reduction: bool = False) -> tuple[bool, str]:
        """Check if position modification is allowed."""
        if not self.enabled or not self.circuit_engine:
            return True, "protection_disabled"
        
        return self.circuit_engine.can_modify_position(is_reduction)
    
    def get_effective_size_multiplier(self) -> float:
        """Get current effective size multiplier."""
        if not self.enabled or not self.circuit_engine:
            return 1.0
        
        return self.circuit_engine.get_effective_size_mult()
    
    def get_effective_leverage_cap(self) -> Optional[float]:
        """Get current effective leverage cap."""
        if not self.enabled or not self.circuit_engine:
            return None
        
        return self.circuit_engine.get_effective_leverage_cap()
    
    def force_level(self, level: str, reason: str = "manual") -> bool:
        """Force circuit to specific level (for testing/manual override)."""
        if not self.enabled or not self.circuit_engine:
            return False
        
        try:
            circuit_level = CircuitLevel(level)
            self.circuit_engine.force_level(circuit_level, reason)
            return True
        except ValueError:
            logger.error(f"[ExtremeEventProtection] Invalid level: {level}")
            return False
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get comprehensive telemetry."""
        if not self.enabled:
            return {"enabled": False}
        
        base_telemetry = {
            "enabled": True,
            "symbol": self.symbol,
            "last_update_ms": self.last_update_ms
        }
        
        if self.circuit_engine:
            base_telemetry["circuit"] = self.circuit_engine.get_telemetry()
        
        if self.execution_adapter:
            base_telemetry["execution"] = {
                "flattener_status": self.execution_adapter.get_flattener_status(),
                "exit_metrics": self.execution_adapter.get_exit_metrics()
            }
        
        if self.shock_detector:
            base_telemetry["shock_detector"] = self.shock_detector.get_audit_summary()
        
        return base_telemetry
    
    # Internal callback handlers
    def _handle_cancel_orders(self, params: Dict[str, Any]) -> None:
        """Handle order cancellation callback."""
        logger.warning(f"[ExtremeEventProtection] Cancelling orders: {params.get('reason', 'unknown')}")
        
        # Execute cancellation if execution adapter available
        if self.execution_adapter:
            asyncio.create_task(self.execution_adapter.cancel_all_resting_orders(
                symbol=self.symbol,
                reason=params.get("reason", "circuit_breaker")
            ))
        
        # Notify external callbacks
        self._notify_callbacks("cancel_orders", params)
    
    def _handle_flatten_positions(self, params: Dict[str, Any]) -> None:
        """Handle position flattening callback."""
        logger.error(f"[ExtremeEventProtection] Flattening positions: {params}")
        
        # Execute flattening if execution adapter available
        if self.execution_adapter and self.circuit_engine:
            circuit_level = self.circuit_engine.state.level
            threat_score = self._get_latest_threat_score()
            
            asyncio.create_task(self.execution_adapter.flatten_all_positions(
                circuit_level=circuit_level,
                threat_score=threat_score,
                symbols=[self.symbol] if self.symbol else None
            ))
        
        # Notify external callbacks
        self._notify_callbacks("flatten_positions", params)
    
    def _handle_size_change(self, params: Dict[str, Any]) -> None:
        """Handle size multiplier change callback."""
        logger.info(f"[ExtremeEventProtection] Size multiplier changed: {params}")
        
        # Notify external callbacks
        self._notify_callbacks("size_mult_change", params)
    
    def _handle_alert(self, params: Dict[str, Any]) -> None:
        """Handle alert generation callback."""
        if not self.alerts or not self.telegram_settings:
            return
        
        # Send async alert
        asyncio.create_task(self._send_alert_async(params))
    
    async def _send_alert_async(self, params: Dict[str, Any]) -> None:
        """Send alert asynchronously."""
        try:
            if not self.alerts or not self.circuit_engine:
                return
            
            level_str = params.get("level", "unknown")
            circuit_level = CircuitLevel(level_str)
            
            threat_score = self._get_latest_threat_score()
            
            success = await self.alerts.send_circuit_alert(
                level=circuit_level,
                state=self.circuit_engine.state,
                threat_score=threat_score,
                settings=self.telegram_settings
            )
            
            if success:
                self._notify_callbacks("alert_sent", params)
                
        except Exception as e:
            logger.error(f"[ExtremeEventProtection] Alert sending failed: {e}")
    
    def _notify_callbacks(self, event_type: str, params: Dict[str, Any]) -> None:
        """Notify registered callbacks."""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(params)
            except Exception as e:
                logger.error(f"[ExtremeEventProtection] Callback error for {event_type}: {e}")


# Factory function for easy creation
def create_extreme_event_protection(settings: ExtremeEventProtectionSettings,
                                   symbol: str = "",
                                   telegram_settings: Optional[Dict] = None) -> ExtremeEventProtectionManager:
    """Factory function to create extreme event protection manager."""
    return ExtremeEventProtectionManager(settings, symbol, telegram_settings)


__all__ = ["ExtremeEventProtectionManager", "ExtremeEventStatus", "create_extreme_event_protection"]
