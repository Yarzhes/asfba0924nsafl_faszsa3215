"""Extreme Event Protection - Telegram Alerts (Sprint 65)

Concise Telegram messages with trigger details and countdown timers.
Templates for different circuit breaker levels.
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..transport.telegram import send_message
from .circuit_breaker_engine import CircuitLevel, CircuitState
from .shock_detector import ShockTrigger


@dataclass
class AlertConfig:
    """Configuration for circuit breaker alerts."""
    enabled: bool = True
    rate_limit_sec: int = 30  # Min seconds between alerts of same type
    max_triggers_shown: int = 3  # Max triggers to show in message
    include_countdown: bool = True
    include_technical_details: bool = True


class CircuitBreakerAlerts:
    """Handles Telegram notifications for circuit breaker events."""
    
    def __init__(self, config: AlertConfig, symbol: str = ""):
        self.config = config
        self.symbol = symbol
        self.last_alert_times: Dict[str, int] = {}  # alert_type -> timestamp_ms
    
    def should_send_alert(self, alert_type: str, timestamp_ms: Optional[int] = None) -> bool:
        """Check if enough time has passed since last alert of this type."""
        if not self.config.enabled:
            return False
        
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        
        last_time = self.last_alert_times.get(alert_type, 0)
        elapsed_sec = (timestamp_ms - last_time) / 1000.0
        
        return elapsed_sec >= self.config.rate_limit_sec
    
    def format_shock_watch_alert(self, triggers: List[ShockTrigger], 
                                threat_score: float) -> str:
        """Format warning level alert (Shock Watch)."""
        # Get top triggers to show
        top_triggers = sorted(triggers, key=lambda t: t.z_score or t.value, reverse=True)
        top_triggers = top_triggers[:self.config.max_triggers_shown]
        
        # Build trigger summary
        trigger_parts = []
        for trigger in top_triggers:
            if trigger.z_score is not None:
                trigger_parts.append(f"{trigger.z_score:.1f}σ")
            else:
                if "PCT" in trigger.type:
                    trigger_parts.append(f"{trigger.value:.1%}")
                elif "BPS" in trigger.type:
                    trigger_parts.append(f"{trigger.value:.0f}bps")
                else:
                    trigger_parts.append(f"{trigger.value:.2f}")
        
        trigger_summary = "/".join(trigger_parts)
        
        # Main message
        msg = f"⚠️ **Shock Watch** | {self.symbol}\n"
        msg += f"🎯 **{trigger_summary}** | Threat: {threat_score:.1f}\n"
        
        if self.config.include_technical_details and triggers:
            details = []
            for trigger in top_triggers:
                short_name = trigger.type.replace("_", "").replace("SIG", "σ")
                details.append(short_name)
            msg += f"📊 {', '.join(details)}\n"
        
        msg += f"🔧 **Actions**: Size halved, wider stops\n"
        
        return msg
    
    def format_derisk_alert(self, triggers: List[ShockTrigger], 
                           threat_score: float) -> str:
        """Format de-risk level alert."""
        # Key metrics from triggers
        vpin_trigger = next((t for t in triggers if "VPIN" in t.type), None)
        lambda_trigger = next((t for t in triggers if "LAMBDA" in t.type), None)
        ret_trigger = next((t for t in triggers if "RET_" in t.type), None)
        
        msg = f"🛡️ **De-risk Mode** | {self.symbol}\n"
        
        # Show key metrics
        metrics = []
        if ret_trigger and ret_trigger.z_score:
            metrics.append(f"Ret {ret_trigger.z_score:.1f}σ")
        if vpin_trigger:
            pctl = int(vpin_trigger.value * 100) if vpin_trigger.value <= 1 else int(vpin_trigger.value)
            metrics.append(f"VPIN p{pctl}")
        if lambda_trigger and lambda_trigger.z_score:
            metrics.append(f"λ z={lambda_trigger.z_score:.1f}")
        
        if metrics:
            msg += f"📈 **{' | '.join(metrics)}**\n"
        
        msg += f"🚫 **New entries paused** | Resting orders cancelled\n"
        msg += f"💹 Reductions & hedges only\n"
        
        return msg
    
    def format_flatten_alert(self, triggers: List[ShockTrigger], 
                            threat_score: float,
                            participation_cap: float = 0.1) -> str:
        """Format flatten all positions alert."""
        # Find the most severe trigger
        severity_order = ["RET_", "RV_", "VPIN_", "VENUE_", "SPREAD_"]
        main_trigger = None
        for prefix in severity_order:
            main_trigger = next((t for t in triggers if t.type.startswith(prefix)), None)
            if main_trigger:
                break
        
        msg = f"🛑 **Flatten All** | {self.symbol}\n"
        
        if main_trigger:
            if "RET_" in main_trigger.type and main_trigger.z_score:
                window = main_trigger.type.split("_")[1] if "_" in main_trigger.type else "?"
                msg += f"💥 **{main_trigger.z_score:.1f}σ move in {window}** detected\n"
            elif "VPIN_" in main_trigger.type:
                pctl = int(main_trigger.value * 100) if main_trigger.value <= 1 else int(main_trigger.value)
                msg += f"🔴 **Toxic flow spike** | VPIN p{pctl}\n"
            elif "VENUE_" in main_trigger.type:
                msg += f"⚠️ **Venue degradation** | Health {main_trigger.value:.1%}\n"
            else:
                msg += f"📊 **Market stress** | Multiple triggers\n"
        
        participation_pct = int(participation_cap * 100)
        msg += f"🎯 **Safe exit** | TWAP {participation_pct}% participation rate\n"
        msg += f"⏱️ Positions closing...\n"
        
        return msg
    
    def format_halt_alert(self, triggers: List[ShockTrigger], 
                         threat_score: float) -> str:
        """Format trading halt alert."""
        msg = f"🔴 **TRADING HALT** | {self.symbol}\n"
        msg += f"⚡ **EXTREME EVENT DETECTED**\n"
        
        # Count trigger types
        ret_triggers = [t for t in triggers if "RET_" in t.type]
        venue_triggers = [t for t in triggers if "VENUE" in t.type]
        depeg_triggers = [t for t in triggers if "DEPEG" in t.type]
        
        alerts = []
        if ret_triggers:
            max_sigma = max(t.z_score or 0 for t in ret_triggers)
            alerts.append(f"🚨 {max_sigma:.1f}σ price spike")
        if venue_triggers:
            alerts.append("⚠️ Venue failure")
        if depeg_triggers:
            alerts.append("💸 Stablecoin depeg")
        if len(triggers) - len(ret_triggers) - len(venue_triggers) - len(depeg_triggers) > 0:
            alerts.append("📊 Multiple stress signals")
        
        if alerts:
            msg += f"\n".join(f"• {alert}" for alert in alerts[:3]) + "\n"
        
        msg += f"🛑 **All trading suspended** until stability returns\n"
        
        return msg
    
    def format_resume_alert(self, recovery_stage: int = 1, 
                           total_stages: int = 4,
                           size_mult: float = 0.5,
                           stability_duration_sec: int = 180) -> str:
        """Format trading resume alert."""
        msg = f"✅ **Resumed** | {self.symbol}\n"
        msg += f"📊 **Metrics normalized** for {stability_duration_sec}s\n"
        
        if recovery_stage > 1:
            msg += f"🔄 **Stage {recovery_stage}/{total_stages}** | Size {size_mult:.1%}\n"
            if recovery_stage < total_stages:
                msg += f"⏭️ Next stage in 5 bars\n"
            else:
                msg += f"🎯 **Full capacity restored**\n"
        else:
            msg += f"🎯 **Gradual re-risking** | {total_stages} stages\n"
            msg += f"📈 Current size: {size_mult:.1%}\n"
        
        return msg
    
    def format_countdown_update(self, countdown_bars: int, 
                               current_level: str,
                               threat_score: float) -> str:
        """Format countdown update (sent less frequently)."""
        time_est = countdown_bars * 5  # Assume 5sec bars
        time_unit = "min" if time_est >= 60 else "sec"
        time_val = time_est // 60 if time_est >= 60 else time_est
        
        level_emoji = {
            "warn": "⚠️",
            "derisk": "🛡️", 
            "flatten": "🛑",
            "halt": "🔴"
        }.get(current_level, "⚪")
        
        msg = f"{level_emoji} **{current_level.upper()}** | {self.symbol}\n"
        msg += f"⏱️ **Resume in**: ~{time_val}{time_unit} ({countdown_bars} bars)\n"
        msg += f"📊 Threat: {threat_score:.1f} (monitoring...)\n"
        
        return msg
    
    async def send_circuit_alert(self, level: CircuitLevel, 
                                state: CircuitState,
                                threat_score: float,
                                settings: Dict[str, Any],
                                countdown_info: Optional[Dict] = None) -> bool:
        """Send appropriate alert for circuit level change."""
        timestamp_ms = int(time.time() * 1000)
        alert_type = f"circuit_{level.value}"
        
        # Check rate limiting
        if not self.should_send_alert(alert_type, timestamp_ms):
            return False
        
        # Format message based on level
        if level == CircuitLevel.WARN:
            message = self.format_shock_watch_alert(state.triggers, threat_score)
        elif level == CircuitLevel.DERISK:
            message = self.format_derisk_alert(state.triggers, threat_score)
        elif level == CircuitLevel.FLATTEN:
            message = self.format_flatten_alert(state.triggers, threat_score)
        elif level == CircuitLevel.HALT:
            message = self.format_halt_alert(state.triggers, threat_score)
        else:
            return False  # Don't alert for normal level
        
        try:
            await send_message(message, settings)
            self.last_alert_times[alert_type] = timestamp_ms
            return True
        except Exception as e:
            # Log error but don't propagate (alerts are non-critical)
            import logging
            logging.error(f"Failed to send circuit alert: {e}")
            return False
    
    async def send_resume_alert(self, recovery_info: Dict[str, Any], 
                               settings: Dict[str, Any]) -> bool:
        """Send trading resume alert."""
        timestamp_ms = int(time.time() * 1000)
        alert_type = "circuit_resume"
        
        if not self.should_send_alert(alert_type, timestamp_ms):
            return False
        
        message = self.format_resume_alert(
            recovery_stage=recovery_info.get("recovery_stage", 1),
            total_stages=recovery_info.get("total_stages", 4),
            size_mult=recovery_info.get("current_size_mult", 1.0),
            stability_duration_sec=recovery_info.get("stability_duration_sec", 180)
        )
        
        try:
            await send_message(message, settings)
            self.last_alert_times[alert_type] = timestamp_ms
            return True
        except Exception as e:
            import logging
            logging.error(f"Failed to send resume alert: {e}")
            return False
    
    async def send_countdown_update(self, countdown_info: Dict[str, Any],
                                   threat_score: float,
                                   settings: Dict[str, Any]) -> bool:
        """Send countdown update (rate limited to avoid spam)."""
        timestamp_ms = int(time.time() * 1000)
        alert_type = "circuit_countdown"
        
        # More conservative rate limiting for countdown updates
        if not self.should_send_alert(alert_type, timestamp_ms):
            return False
        
        # Only send countdown if substantial time remaining
        countdown_bars = countdown_info.get("countdown_bars", 0)
        if countdown_bars < 5:  # Don't spam near the end
            return False
        
        message = self.format_countdown_update(
            countdown_bars=countdown_bars,
            current_level=countdown_info.get("level", "unknown"),
            threat_score=threat_score
        )
        
        try:
            await send_message(message, settings)
            self.last_alert_times[alert_type] = timestamp_ms
            return True
        except Exception as e:
            import logging
            logging.error(f"Failed to send countdown alert: {e}")
            return False


__all__ = ["CircuitBreakerAlerts", "AlertConfig"]
