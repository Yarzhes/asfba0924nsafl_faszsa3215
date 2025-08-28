"""Extreme Event Protection - Circuit Breaker Engine (Sprint 65)

Tiered circuit breakers with hysteresis and gradual recovery.
Handles: Warn → De-risk → Flatten → Halt with cooldown logic.
"""
from __future__ import annotations
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from loguru import logger

from .shock_detector import ShockDetector, ShockFeatures, ShockTrigger, ShockConfig


class CircuitLevel(Enum):
    """Circuit breaker severity levels."""
    NORMAL = 0
    WARN = 1
    DERISK = 2
    FLATTEN = 3
    HALT = 4


@dataclass
class CircuitAction:
    """Specific action to take at circuit level."""
    level: CircuitLevel
    size_mult: float  # Size multiplier (0 = no new positions, 1 = normal)
    leverage_cap: Optional[float] = None  # Max leverage allowed
    cancel_resting: bool = False  # Cancel existing resting orders
    flatten_positions: bool = False  # Close all positions
    block_new_entries: bool = False  # Block new entry orders
    allow_reductions: bool = True  # Allow position reductions
    preferred_exit_style: str = "passive"  # passive, twap, market
    participation_cap: float = 0.1  # Max participation rate for exits


@dataclass
class CircuitPolicy:
    """Circuit breaker policy configuration."""
    # Entry thresholds (higher values = easier to trigger)
    warn_threshold: float = 1.0
    derisk_threshold: float = 2.0  
    flatten_threshold: float = 3.0
    halt_threshold: float = 4.0
    
    # Exit thresholds (lower values = easier to recover, hysteresis)
    warn_exit_threshold: float = 0.5
    derisk_exit_threshold: float = 1.0
    flatten_exit_threshold: float = 1.5
    halt_exit_threshold: float = 2.0
    
    # Cooldown requirements (bars/seconds to stay below exit threshold)
    warn_cooldown_bars: int = 3
    derisk_cooldown_bars: int = 5
    flatten_cooldown_bars: int = 10
    halt_cooldown_bars: int = 20
    
    # Recovery staging (gradual re-risking)
    enable_staged_recovery: bool = True
    recovery_stages: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    recovery_stage_bars: int = 5  # Bars between stages
    
    # Actions per level
    actions: Dict[CircuitLevel, CircuitAction] = field(default_factory=lambda: {
        CircuitLevel.NORMAL: CircuitAction(
            level=CircuitLevel.NORMAL,
            size_mult=1.0
        ),
        CircuitLevel.WARN: CircuitAction(
            level=CircuitLevel.WARN,
            size_mult=0.5,
            leverage_cap=5.0,
            preferred_exit_style="passive"
        ),
        CircuitLevel.DERISK: CircuitAction(
            level=CircuitLevel.DERISK,
            size_mult=0.0,
            leverage_cap=3.0,
            cancel_resting=True,
            block_new_entries=True,
            preferred_exit_style="twap",
            participation_cap=0.05
        ),
        CircuitLevel.FLATTEN: CircuitAction(
            level=CircuitLevel.FLATTEN,
            size_mult=0.0,
            leverage_cap=1.0,
            cancel_resting=True,
            flatten_positions=True,
            block_new_entries=True,
            preferred_exit_style="twap",
            participation_cap=0.1
        ),
        CircuitLevel.HALT: CircuitAction(
            level=CircuitLevel.HALT,
            size_mult=0.0,
            cancel_resting=True,
            flatten_positions=True,
            block_new_entries=True,
            preferred_exit_style="market",
            participation_cap=0.2
        )
    })


@dataclass
class CircuitState:
    """Current circuit breaker state."""
    level: CircuitLevel = CircuitLevel.NORMAL
    triggered_at_ms: int = 0
    reason_codes: List[str] = field(default_factory=list)
    triggers: List[ShockTrigger] = field(default_factory=list)
    
    # Cooldown tracking
    cooldown_bars_left: int = 0
    stability_bars: int = 0  # Consecutive bars below exit threshold
    
    # Recovery staging
    recovery_stage: int = 0  # 0 = not in recovery, 1-N = stage number
    recovery_stage_bars_left: int = 0
    
    # Effective controls
    size_mult_current: float = 1.0
    leverage_cap_current: Optional[float] = None
    
    # Audit trail
    state_changes: deque = field(default_factory=lambda: deque(maxlen=100))
    last_update_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class CircuitBreakerEngine:
    """Tiered circuit breaker system with hysteresis and staged recovery."""
    
    def __init__(self, 
                 shock_detector: ShockDetector,
                 policy: CircuitPolicy,
                 symbol: str = ""):
        self.shock_detector = shock_detector
        self.policy = policy
        self.symbol = symbol
        self.state = CircuitState()
        
        # Metrics tracking
        self.metrics_history: deque = deque(maxlen=500)
        self.stability_tracker: deque = deque(maxlen=50)  # Track stability metrics
        
        # Callbacks for actions
        self.action_callbacks: Dict[str, Callable] = {}
    
    def register_callback(self, action_type: str, callback: Callable) -> None:
        """Register callback for specific circuit actions.
        
        Args:
            action_type: One of 'cancel_orders', 'flatten_positions', 'size_change', 'alert'
            callback: Function to call when action is triggered
        """
        self.action_callbacks[action_type] = callback
    
    def update(self, 
               features: Optional[ShockFeatures] = None,
               timestamp_ms: Optional[int] = None,
               bar_close: bool = False) -> CircuitState:
        """Update circuit breaker state based on current conditions.
        
        Args:
            features: Current shock features (if None, will compute from detector)
            timestamp_ms: Current timestamp
            bar_close: Whether this is a bar close (for cooldown counting)
            
        Returns:
            Updated circuit state
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        
        self.state.last_update_ms = timestamp_ms
        
        # Get shock detection results
        shock_level, triggers = self.shock_detector.detect_shocks(features, timestamp_ms)
        
        # Map string level to enum
        level_map = {
            "normal": CircuitLevel.NORMAL,
            "warn": CircuitLevel.WARN,
            "derisk": CircuitLevel.DERISK,
            "flatten": CircuitLevel.FLATTEN,
            "halt": CircuitLevel.HALT
        }
        shock_circuit_level = level_map.get(shock_level, CircuitLevel.NORMAL)
        
        # Compute composite threat score from triggers
        threat_score = self._compute_threat_score(triggers)
        
        # Store metrics
        self.metrics_history.append((timestamp_ms, threat_score, shock_level, len(triggers)))
        
        # Determine target level based on threat score
        target_level = self._determine_target_level(threat_score)
        
        # Apply hysteresis logic
        new_level = self._apply_hysteresis(target_level, threat_score, bar_close)
        
        # Check if level changed
        level_changed = new_level != self.state.level
        if level_changed:
            self._transition_to_level(new_level, triggers, timestamp_ms)
        
        # Update cooldowns and recovery if bar close
        if bar_close:
            self._update_cooldowns_and_recovery(threat_score)
        
        # Apply current action and update effective controls
        self._apply_current_action()
        
        return self.state
    
    def _compute_threat_score(self, triggers: List[ShockTrigger]) -> float:
        """Compute composite threat score from triggers."""
        if not triggers:
            return 0.0
        
        # Weight different trigger types
        weights = {
            'RET_': 2.0,  # Return spikes are high priority
            'RV_': 1.5,   # Realized vol spikes
            'SPREAD_': 1.2, # Spread deterioration  
            'DEPTH_': 1.0,  # Depth collapse
            'VPIN_': 1.8,   # Toxic flow
            'LAMBDA_': 1.5, # Market impact
            'OI_': 1.0,     # OI dumps
            'FUNDING_': 0.8, # Funding swings
            'VENUE_': 2.0,   # Venue issues
            'DEPEG_': 1.5    # Stablecoin depeg
        }
        
        total_score = 0.0
        for trigger in triggers:
            # Find weight for this trigger type
            weight = 1.0
            for prefix, w in weights.items():
                if trigger.type.startswith(prefix):
                    weight = w
                    break
            
            # Score based on how far above threshold
            if trigger.z_score is not None:
                excess = trigger.z_score - trigger.threshold if trigger.threshold > 0 else trigger.z_score
            else:
                excess = (trigger.value - trigger.threshold) / max(trigger.threshold, 1e-6)
            
            trigger_score = weight * max(1.0, excess)
            total_score += trigger_score
        
        return total_score
    
    def _determine_target_level(self, threat_score: float) -> CircuitLevel:
        """Determine target circuit level based on threat score."""
        if threat_score >= self.policy.halt_threshold:
            return CircuitLevel.HALT
        elif threat_score >= self.policy.flatten_threshold:
            return CircuitLevel.FLATTEN
        elif threat_score >= self.policy.derisk_threshold:
            return CircuitLevel.DERISK
        elif threat_score >= self.policy.warn_threshold:
            return CircuitLevel.WARN
        else:
            return CircuitLevel.NORMAL
    
    def _apply_hysteresis(self, target_level: CircuitLevel, threat_score: float, 
                         bar_close: bool) -> CircuitLevel:
        """Apply hysteresis logic to prevent oscillation."""
        current = self.state.level
        
        # Moving to higher severity (easier)
        if target_level.value > current.value:
            return target_level
        
        # Moving to lower severity (requires hysteresis)
        if target_level.value < current.value:
            # Check if we're below exit threshold
            exit_threshold = self._get_exit_threshold(current)
            if threat_score <= exit_threshold:
                # Track stability
                if bar_close:
                    self.state.stability_bars += 1
                
                # Check if we've been stable long enough
                required_stability = self._get_required_stability_bars(current)
                if self.state.stability_bars >= required_stability:
                    return target_level
            else:
                # Reset stability counter if above exit threshold
                if bar_close:
                    self.state.stability_bars = 0
        
        # No change
        return current
    
    def _get_exit_threshold(self, level: CircuitLevel) -> float:
        """Get exit threshold for given level."""
        if level == CircuitLevel.WARN:
            return self.policy.warn_exit_threshold
        elif level == CircuitLevel.DERISK:
            return self.policy.derisk_exit_threshold
        elif level == CircuitLevel.FLATTEN:
            return self.policy.flatten_exit_threshold
        elif level == CircuitLevel.HALT:
            return self.policy.halt_exit_threshold
        return 0.0
    
    def _get_required_stability_bars(self, level: CircuitLevel) -> int:
        """Get required stability bars for level."""
        if level == CircuitLevel.WARN:
            return self.policy.warn_cooldown_bars
        elif level == CircuitLevel.DERISK:
            return self.policy.derisk_cooldown_bars
        elif level == CircuitLevel.FLATTEN:
            return self.policy.flatten_cooldown_bars
        elif level == CircuitLevel.HALT:
            return self.policy.halt_cooldown_bars
        return 0
    
    def _transition_to_level(self, new_level: CircuitLevel, 
                           triggers: List[ShockTrigger], 
                           timestamp_ms: int) -> None:
        """Transition to new circuit level."""
        old_level = self.state.level
        
        # Log the transition
        reason_codes = [t.type for t in triggers]
        
        logger.warning(
            f"[Circuit] {self.symbol} {old_level.name.upper()} → {new_level.name.upper()} "
            f"| Triggers: {len(triggers)} | Reasons: {', '.join(reason_codes[:3])}"
        )
        
        # Update state
        self.state.level = new_level
        self.state.triggered_at_ms = timestamp_ms
        self.state.reason_codes = reason_codes
        self.state.triggers = triggers.copy()
        self.state.stability_bars = 0
        
        # Reset recovery if moving to higher severity
        if new_level.value > old_level.value:
            self.state.recovery_stage = 0
            self.state.recovery_stage_bars_left = 0
        # Initialize staged recovery when moving from high severity to normal
        elif (new_level == CircuitLevel.NORMAL and 
              old_level.value >= CircuitLevel.DERISK.value and
              self.policy.enable_staged_recovery):
            self.state.recovery_stage = 1  # Start at first recovery stage
            self.state.recovery_stage_bars_left = self.policy.recovery_stage_bars
        
        # Record state change
        change_record = {
            'timestamp_ms': timestamp_ms,
            'from_level': old_level.name.lower(),
            'to_level': new_level.name.lower(),
            'trigger_count': len(triggers),
            'reason_codes': reason_codes[:5]  # Limit for memory
        }
        self.state.state_changes.append(change_record)
        
        # Execute callbacks for significant transitions
        if new_level in [CircuitLevel.DERISK, CircuitLevel.FLATTEN, CircuitLevel.HALT]:
            self._execute_action_callback('alert', {
                'level': new_level.name.lower(),
                'triggers': triggers,
                'symbol': self.symbol
            })
    
    def _update_cooldowns_and_recovery(self, threat_score: float) -> None:
        """Update cooldown timers and recovery staging."""
        # Update cooldown bars
        if self.state.cooldown_bars_left > 0:
            self.state.cooldown_bars_left -= 1
        
        # Handle staged recovery
        if (self.policy.enable_staged_recovery and 
            self.state.level == CircuitLevel.NORMAL and 
            self.state.recovery_stage > 0):
            
            if self.state.recovery_stage_bars_left > 0:
                self.state.recovery_stage_bars_left -= 1
            else:
                # Advance to next recovery stage
                if self.state.recovery_stage < len(self.policy.recovery_stages):
                    self.state.recovery_stage += 1
                    self.state.recovery_stage_bars_left = self.policy.recovery_stage_bars
                    
                    logger.info(f"[Circuit] {self.symbol} Recovery stage {self.state.recovery_stage}/{len(self.policy.recovery_stages)}")
                    
                    # Check if recovery is complete
                    if self.state.recovery_stage >= len(self.policy.recovery_stages):
                        logger.info(f"[Circuit] {self.symbol} Recovery complete - returning to normal operations")
                        self.state.recovery_stage = 0  # End recovery mode
    
    def _apply_current_action(self) -> None:
        """Apply the action for current circuit level."""
        action = self.policy.actions.get(self.state.level)
        if not action:
            return
        
        # Determine effective size multiplier
        if self.policy.enable_staged_recovery and self.state.recovery_stage > 0:
            # Apply staged recovery
            stage_idx = min(self.state.recovery_stage - 1, len(self.policy.recovery_stages) - 1)
            recovery_mult = self.policy.recovery_stages[stage_idx]
            self.state.size_mult_current = action.size_mult * recovery_mult
        else:
            self.state.size_mult_current = action.size_mult
        
        # Apply leverage cap
        self.state.leverage_cap_current = action.leverage_cap
        
        # Execute immediate actions if level just changed
        if action.cancel_resting:
            self._execute_action_callback('cancel_orders', {'reason': 'circuit_breaker'})
        
        if action.flatten_positions:
            self._execute_action_callback('flatten_positions', {
                'style': action.preferred_exit_style,
                'participation_cap': action.participation_cap
            })
        
        # Notify of size multiplier changes
        self._execute_action_callback('size_change', {
            'size_mult': self.state.size_mult_current,
            'level': self.state.level.name.lower()
        })
    
    def _execute_action_callback(self, action_type: str, params: Dict[str, Any]) -> None:
        """Execute registered callback for action type."""
        callback = self.action_callbacks.get(action_type)
        if callback:
            try:
                callback(params)
            except Exception as e:
                logger.error(f"[Circuit] Callback error for {action_type}: {e}")
    
    def force_level(self, level: CircuitLevel, reason: str = "manual") -> None:
        """Force circuit to specific level (for testing/manual override)."""
        timestamp_ms = int(time.time() * 1000)
        
        # Create dummy trigger for audit trail
        trigger = ShockTrigger("MANUAL_OVERRIDE", 1.0, 0.0, None, timestamp_ms)
        
        self._transition_to_level(level, [trigger], timestamp_ms)
        self._apply_current_action()
        
        logger.warning(f"[Circuit] {self.symbol} Force level: {level.name.lower()} | Reason: {reason}")
    
    def can_enter_position(self, side: str = "long") -> Tuple[bool, str]:
        """Check if new position entry is allowed."""
        action = self.policy.actions.get(self.state.level)
        if not action:
            return True, "no_policy"
        
        if action.block_new_entries:
            return False, f"circuit_{self.state.level.name.lower()}"
        
        if self.state.size_mult_current <= 0:
            return False, f"size_mult_zero"
        
        return True, "allowed"
    
    def can_modify_position(self, is_reduction: bool = False) -> Tuple[bool, str]:
        """Check if position modification is allowed."""
        action = self.policy.actions.get(self.state.level)
        if not action:
            return True, "no_policy"
        
        # Reductions generally allowed even in strict modes
        if is_reduction and action.allow_reductions:
            return True, "reduction_allowed"
        
        # Increases not allowed if blocking entries
        if not is_reduction and action.block_new_entries:
            return False, f"circuit_{self.state.level.name.lower()}"
        
        return True, "allowed"
    
    def get_effective_size_mult(self) -> float:
        """Get current effective size multiplier."""
        return self.state.size_mult_current
    
    def get_effective_leverage_cap(self) -> Optional[float]:
        """Get current effective leverage cap."""
        return self.state.leverage_cap_current
    
    def get_preferred_exit_style(self) -> str:
        """Get preferred exit style for current level."""
        action = self.policy.actions.get(self.state.level)
        return action.preferred_exit_style if action else "passive"
    
    def get_resume_countdown(self) -> Dict[str, Any]:
        """Get countdown information for resume."""
        if self.state.level == CircuitLevel.NORMAL:
            return {"status": "normal", "countdown_bars": 0}
        
        required_bars = self._get_required_stability_bars(self.state.level)
        remaining_bars = max(0, required_bars - self.state.stability_bars)
        
        status = "counting_down" if remaining_bars > 0 else "ready_to_resume"
        
        recovery_info = {}
        if (self.policy.enable_staged_recovery and 
            self.state.level == CircuitLevel.NORMAL and 
            self.state.recovery_stage > 0):
            recovery_info = {
                "recovery_stage": self.state.recovery_stage,
                "total_stages": len(self.policy.recovery_stages),
                "stage_bars_left": self.state.recovery_stage_bars_left,
                "current_size_mult": self.state.size_mult_current
            }
        
        return {
            "status": status,
            "level": self.state.level.name.lower(),
            "countdown_bars": remaining_bars,
            "stability_bars": self.state.stability_bars,
            "required_bars": required_bars,
            "recovery": recovery_info
        }
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get comprehensive telemetry for monitoring."""
        # Recent metrics summary
        recent_metrics = list(self.metrics_history)[-20:] if self.metrics_history else []
        
        threat_scores = [m[1] for m in recent_metrics]
        current_threat = threat_scores[-1] if threat_scores else 0.0
        avg_threat = np.mean(threat_scores) if threat_scores else 0.0
        max_threat = max(threat_scores) if threat_scores else 0.0
        
        # Shock detector audit
        shock_audit = self.shock_detector.get_audit_summary(300.0)  # 5 min lookback
        
        return {
            # Current state
            "circuit_state": self.state.level.name.lower(),
            "size_mult_current": self.state.size_mult_current,
            "leverage_cap_current": self.state.leverage_cap_current,
            "triggered_at_ms": self.state.triggered_at_ms,
            "stability_bars": self.state.stability_bars,
            
            # Threat analysis
            "threat_score_current": current_threat,
            "threat_score_avg_20": avg_threat,
            "threat_score_max_20": max_threat,
            
            # Triggers
            "reason_codes": self.state.reason_codes,
            "trigger_count": len(self.state.triggers),
            
            # Recovery status
            "resume_countdown": self.get_resume_countdown(),
            
            # Shock detector stats
            "shock_audit": shock_audit,
            
            # Audit trail
            "recent_state_changes": len(self.state.state_changes),
            "last_update_ms": self.state.last_update_ms
        }


__all__ = ["CircuitBreakerEngine", "CircuitPolicy", "CircuitState", "CircuitLevel", "CircuitAction"]
