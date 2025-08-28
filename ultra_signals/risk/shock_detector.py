"""Extreme Event Protection - Shock Detector (Sprint 65)

Multi-sigma price/vol spikes with microstructure corroboration.
Real-time detection of black swan and flash crash events.
"""
from __future__ import annotations
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger


@dataclass
class ShockTrigger:
    """Represents a shock detection trigger."""
    type: str  # e.g., 'RET_5S_6SIG', 'SPREAD_3SIG', 'VPIN_95PCTL'
    value: float
    threshold: float
    z_score: Optional[float] = None
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class ShockFeatures:
    """Feature snapshot for shock detection."""
    shock_ret_z: float = 0.0
    rv_short_z: float = 0.0
    spread_z: float = 0.0
    depth_drop_pct: float = 0.0
    vpin_pctl: float = 0.0
    lambda_z: float = 0.0
    # Additional features
    oi_dump_pct: Optional[float] = None
    funding_swing_bps: Optional[float] = None
    venue_health_score: float = 1.0
    stablecoin_depeg_bps: Optional[float] = None
    # Raw values for context
    return_pct: float = 0.0
    realized_vol: float = 0.0
    spread_bps: float = 0.0
    top_depth_ratio: float = 1.0


@dataclass
class ShockConfig:
    """Configuration for shock detection thresholds."""
    # Return spike detection (multiple windows)
    return_windows_sec: List[float] = field(default_factory=lambda: [1.0, 2.0, 5.0])
    warn_k_sigma: float = 4.0
    derisk_k_sigma: float = 5.0
    flatten_k_sigma: float = 6.0
    halt_k_sigma: float = 8.0
    
    # Realized volatility thresholds
    rv_horizon_sec: float = 10.0
    rv_warn_z: float = 2.5
    rv_derisk_z: float = 3.0
    rv_flatten_z: float = 4.0
    
    # Order book stress
    spread_warn_z: float = 2.0
    spread_derisk_z: float = 3.0
    depth_warn_drop_pct: float = 0.5  # 50% depth collapse
    depth_derisk_drop_pct: float = 0.7  # 70% depth collapse
    
    # VPIN & Lambda thresholds (percentiles)
    vpin_warn_pctl: float = 0.90
    vpin_derisk_pctl: float = 0.95
    vpin_flatten_pctl: float = 0.98
    lambda_warn_z: float = 2.0
    lambda_derisk_z: float = 3.0
    
    # Derivatives stress
    oi_dump_warn_pct: float = 0.10  # 10% OI drop
    oi_dump_derisk_pct: float = 0.20  # 20% OI drop
    funding_swing_warn_bps: float = 10.0
    funding_swing_derisk_bps: float = 25.0
    
    # Venue/data quality
    venue_health_warn: float = 0.8
    venue_health_derisk: float = 0.6
    stablecoin_depeg_warn_bps: float = 20.0
    stablecoin_depeg_derisk_bps: float = 50.0
    
    # Multi-trigger requirements (N-of-M logic)
    min_triggers_warn: int = 1
    min_triggers_derisk: int = 2
    min_triggers_flatten: int = 2
    min_triggers_halt: int = 3


class ShockDetector:
    """Real-time extreme event detection engine.
    
    Combines multiple signals to detect black swan / flash crash events:
    - Multi-timeframe return spikes
    - Realized volatility jumps
    - Order book deterioration
    - Toxic flow indicators (VPIN, Lambda)
    - Derivatives stress signals
    - Venue/data quality issues
    """
    
    def __init__(self, config: ShockConfig, symbol: str = ""):
        self.config = config
        self.symbol = symbol
        
        # Price/return tracking for multiple windows
        self.price_history: deque = deque(maxlen=1000)  # (timestamp_ms, price)
        self.return_buffers: Dict[float, deque] = {}
        for window_sec in config.return_windows_sec:
            self.return_buffers[window_sec] = deque(maxlen=100)
        
        # Realized volatility tracking
        self.rv_buffer: deque = deque(maxlen=100)
        self.rv_history: deque = deque(maxlen=500)  # for z-score computation
        
        # Order book tracking
        self.spread_history: deque = deque(maxlen=200)
        self.depth_history: deque = deque(maxlen=200)
        
        # VPIN & Lambda tracking
        self.vpin_history: deque = deque(maxlen=500)
        self.lambda_history: deque = deque(maxlen=200)
        
        # Derivatives tracking
        self.oi_history: deque = deque(maxlen=100)
        self.funding_history: deque = deque(maxlen=50)
        
        # State tracking
        self.last_update_ms: int = 0
        self.recent_triggers: deque = deque(maxlen=100)  # Recent trigger history
        
    def update_price(self, timestamp_ms: int, price: float) -> None:
        """Update price history and compute returns for all windows."""
        self.price_history.append((timestamp_ms, price))
        self.last_update_ms = timestamp_ms
        
        # Compute returns for each configured window
        for window_sec in self.config.return_windows_sec:
            window_ms = window_sec * 1000
            cutoff_ms = timestamp_ms - window_ms
            
            # Find price at window start
            start_price = None
            for ts, p in reversed(self.price_history):
                if ts <= cutoff_ms:
                    start_price = p
                    break
            
            if start_price is not None and start_price > 0:
                ret_pct = (price - start_price) / start_price
                self.return_buffers[window_sec].append((timestamp_ms, ret_pct))
    
    def update_realized_vol(self, timestamp_ms: int, rv: float) -> None:
        """Update realized volatility."""
        self.rv_buffer.append((timestamp_ms, rv))
        self.rv_history.append(rv)
        
    def update_orderbook(self, timestamp_ms: int, spread_bps: float, 
                        top_bid_qty: float, top_ask_qty: float, 
                        ref_depth: Optional[float] = None) -> None:
        """Update order book metrics."""
        self.spread_history.append((timestamp_ms, spread_bps))
        
        # Compute depth ratio vs reference (e.g., historical average)
        current_depth = min(top_bid_qty, top_ask_qty)
        if ref_depth is not None and ref_depth > 0:
            depth_ratio = current_depth / ref_depth
        else:
            depth_ratio = 1.0
        self.depth_history.append((timestamp_ms, depth_ratio))
    
    def update_vpin(self, vpin: float, vpin_pctl: float) -> None:
        """Update VPIN metrics."""
        self.vpin_history.append(vpin)
    
    def update_lambda(self, lambda_val: float) -> None:
        """Update lambda (market impact) metric."""
        self.lambda_history.append(lambda_val)
    
    def update_derivatives(self, oi_change_pct: Optional[float] = None, 
                          funding_rate_bps: Optional[float] = None) -> None:
        """Update derivatives stress indicators."""
        if oi_change_pct is not None:
            self.oi_history.append(oi_change_pct)
        if funding_rate_bps is not None:
            self.funding_history.append(funding_rate_bps)
    
    def compute_features(self, venue_health: float = 1.0,
                        stablecoin_depeg_bps: Optional[float] = None) -> ShockFeatures:
        """Compute current shock detection features."""
        features = ShockFeatures()
        
        # Return spike z-scores (max across all windows)
        max_ret_z = 0.0
        max_ret_pct = 0.0
        for window_sec, returns in self.return_buffers.items():
            if len(returns) < 2:
                continue
            
            recent_returns = [r for _, r in returns]
            if len(recent_returns) < 10:
                continue
                
            latest_ret = recent_returns[-1]
            ret_std = np.std(recent_returns[:-1])
            if ret_std > 0:
                ret_z = abs(latest_ret) / ret_std
                if ret_z > max_ret_z:
                    max_ret_z = ret_z
                    max_ret_pct = latest_ret
        
        features.shock_ret_z = max_ret_z
        features.return_pct = max_ret_pct
        
        # Realized vol z-score
        if len(self.rv_history) > 10:
            recent_rv = list(self.rv_history)
            current_rv = recent_rv[-1]
            hist_mean = np.mean(recent_rv[:-1])
            hist_std = np.std(recent_rv[:-1])
            if hist_std > 0:
                features.rv_short_z = (current_rv - hist_mean) / hist_std
            features.realized_vol = current_rv
        
        # Spread z-score
        if len(self.spread_history) > 10:
            recent_spreads = [s for _, s in list(self.spread_history)]
            current_spread = recent_spreads[-1]
            hist_mean = np.mean(recent_spreads[:-1])
            hist_std = np.std(recent_spreads[:-1])
            if hist_std > 0:
                features.spread_z = (current_spread - hist_mean) / hist_std
            features.spread_bps = current_spread
        
        # Depth drop percentage
        if len(self.depth_history) > 1:
            recent_depths = [d for _, d in list(self.depth_history)]
            current_depth = recent_depths[-1]
            features.depth_drop_pct = max(0, 1.0 - current_depth)
            features.top_depth_ratio = current_depth
        
        # VPIN percentile (computed externally, just store)
        if self.vpin_history:
            vpin_vals = list(self.vpin_history)
            current_vpin = vpin_vals[-1] if vpin_vals else 0.0
            if len(vpin_vals) > 1:
                features.vpin_pctl = (np.array(vpin_vals) < current_vpin).mean() * 100
            else:
                # If we only have one value, assume it's already a percentile
                features.vpin_pctl = current_vpin * 100
        
        # Lambda z-score
        if len(self.lambda_history) > 10:
            lambda_vals = list(self.lambda_history)
            current_lambda = lambda_vals[-1]
            hist_mean = np.mean(lambda_vals[:-1])
            hist_std = np.std(lambda_vals[:-1])
            if hist_std > 0:
                features.lambda_z = (current_lambda - hist_mean) / hist_std
        
        # Derivatives stress
        if self.oi_history:
            features.oi_dump_pct = abs(min(self.oi_history[-5:], default=0.0))
        
        if len(self.funding_history) >= 2:
            funding_vals = list(self.funding_history)
            funding_swing = abs(funding_vals[-1] - funding_vals[-2])
            features.funding_swing_bps = funding_swing
        
        # External inputs
        features.venue_health_score = venue_health
        features.stablecoin_depeg_bps = stablecoin_depeg_bps
        
        return features
    
    def detect_shocks(self, features: Optional[ShockFeatures] = None, 
                     timestamp_ms: Optional[int] = None) -> Tuple[str, List[ShockTrigger]]:
        """Detect shock events and return severity level + triggers.
        
        Returns:
            Tuple of (level, triggers) where level is one of:
            'normal', 'warn', 'derisk', 'flatten', 'halt'
        """
        if features is None:
            features = self.compute_features()
        
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        
        triggers: List[ShockTrigger] = []
        
        # Return spike triggers
        for window_sec in self.config.return_windows_sec:
            if features.shock_ret_z >= self.config.halt_k_sigma:
                triggers.append(ShockTrigger(
                    f"RET_{window_sec}S_{self.config.halt_k_sigma}SIG",
                    features.shock_ret_z, self.config.halt_k_sigma,
                    features.shock_ret_z, timestamp_ms
                ))
            elif features.shock_ret_z >= self.config.flatten_k_sigma:
                triggers.append(ShockTrigger(
                    f"RET_{window_sec}S_{self.config.flatten_k_sigma}SIG",
                    features.shock_ret_z, self.config.flatten_k_sigma,
                    features.shock_ret_z, timestamp_ms
                ))
            elif features.shock_ret_z >= self.config.derisk_k_sigma:
                triggers.append(ShockTrigger(
                    f"RET_{window_sec}S_{self.config.derisk_k_sigma}SIG",
                    features.shock_ret_z, self.config.derisk_k_sigma,
                    features.shock_ret_z, timestamp_ms
                ))
            elif features.shock_ret_z >= self.config.warn_k_sigma:
                triggers.append(ShockTrigger(
                    f"RET_{window_sec}S_{self.config.warn_k_sigma}SIG",
                    features.shock_ret_z, self.config.warn_k_sigma,
                    features.shock_ret_z, timestamp_ms
                ))
        
        # Realized volatility triggers
        if features.rv_short_z >= self.config.rv_flatten_z:
            triggers.append(ShockTrigger(
                "RV_4SIG", features.rv_short_z, self.config.rv_flatten_z,
                features.rv_short_z, timestamp_ms
            ))
        elif features.rv_short_z >= self.config.rv_derisk_z:
            triggers.append(ShockTrigger(
                "RV_3SIG", features.rv_short_z, self.config.rv_derisk_z,
                features.rv_short_z, timestamp_ms
            ))
        elif features.rv_short_z >= self.config.rv_warn_z:
            triggers.append(ShockTrigger(
                "RV_2SIG", features.rv_short_z, self.config.rv_warn_z,
                features.rv_short_z, timestamp_ms
            ))
        
        # Spread stress triggers
        if features.spread_z >= self.config.spread_derisk_z:
            triggers.append(ShockTrigger(
                "SPREAD_3SIG", features.spread_z, self.config.spread_derisk_z,
                features.spread_z, timestamp_ms
            ))
        elif features.spread_z >= self.config.spread_warn_z:
            triggers.append(ShockTrigger(
                "SPREAD_2SIG", features.spread_z, self.config.spread_warn_z,
                features.spread_z, timestamp_ms
            ))
        
        # Depth collapse triggers
        if features.depth_drop_pct >= self.config.depth_derisk_drop_pct:
            triggers.append(ShockTrigger(
                "DEPTH_70PCT", features.depth_drop_pct, self.config.depth_derisk_drop_pct,
                None, timestamp_ms
            ))
        elif features.depth_drop_pct >= self.config.depth_warn_drop_pct:
            triggers.append(ShockTrigger(
                "DEPTH_50PCT", features.depth_drop_pct, self.config.depth_warn_drop_pct,
                None, timestamp_ms
            ))
        
        # VPIN toxicity triggers
        if features.vpin_pctl >= self.config.vpin_flatten_pctl:
            triggers.append(ShockTrigger(
                "VPIN_98PCTL", features.vpin_pctl, self.config.vpin_flatten_pctl,
                None, timestamp_ms
            ))
        elif features.vpin_pctl >= self.config.vpin_derisk_pctl:
            triggers.append(ShockTrigger(
                "VPIN_95PCTL", features.vpin_pctl, self.config.vpin_derisk_pctl,
                None, timestamp_ms
            ))
        elif features.vpin_pctl >= self.config.vpin_warn_pctl:
            triggers.append(ShockTrigger(
                "VPIN_90PCTL", features.vpin_pctl, self.config.vpin_warn_pctl,
                None, timestamp_ms
            ))
        
        # Lambda stress triggers
        if features.lambda_z >= self.config.lambda_derisk_z:
            triggers.append(ShockTrigger(
                "LAMBDA_3SIG", features.lambda_z, self.config.lambda_derisk_z,
                features.lambda_z, timestamp_ms
            ))
        elif features.lambda_z >= self.config.lambda_warn_z:
            triggers.append(ShockTrigger(
                "LAMBDA_2SIG", features.lambda_z, self.config.lambda_warn_z,
                features.lambda_z, timestamp_ms
            ))
        
        # Derivatives stress triggers
        if features.oi_dump_pct and features.oi_dump_pct >= self.config.oi_dump_derisk_pct:
            triggers.append(ShockTrigger(
                "OI_DUMP_20PCT", features.oi_dump_pct, self.config.oi_dump_derisk_pct,
                None, timestamp_ms
            ))
        elif features.oi_dump_pct and features.oi_dump_pct >= self.config.oi_dump_warn_pct:
            triggers.append(ShockTrigger(
                "OI_DUMP_10PCT", features.oi_dump_pct, self.config.oi_dump_warn_pct,
                None, timestamp_ms
            ))
        
        if features.funding_swing_bps and features.funding_swing_bps >= self.config.funding_swing_derisk_bps:
            triggers.append(ShockTrigger(
                "FUNDING_25BPS", features.funding_swing_bps, self.config.funding_swing_derisk_bps,
                None, timestamp_ms
            ))
        elif features.funding_swing_bps and features.funding_swing_bps >= self.config.funding_swing_warn_bps:
            triggers.append(ShockTrigger(
                "FUNDING_10BPS", features.funding_swing_bps, self.config.funding_swing_warn_bps,
                None, timestamp_ms
            ))
        
        # Venue health triggers
        if features.venue_health_score <= self.config.venue_health_derisk:
            triggers.append(ShockTrigger(
                "VENUE_DEGRADED", features.venue_health_score, self.config.venue_health_derisk,
                None, timestamp_ms
            ))
        elif features.venue_health_score <= self.config.venue_health_warn:
            triggers.append(ShockTrigger(
                "VENUE_WARN", features.venue_health_score, self.config.venue_health_warn,
                None, timestamp_ms
            ))
        
        # Stablecoin depeg triggers
        if features.stablecoin_depeg_bps and features.stablecoin_depeg_bps >= self.config.stablecoin_depeg_derisk_bps:
            triggers.append(ShockTrigger(
                "DEPEG_50BPS", features.stablecoin_depeg_bps, self.config.stablecoin_depeg_derisk_bps,
                None, timestamp_ms
            ))
        elif features.stablecoin_depeg_bps and features.stablecoin_depeg_bps >= self.config.stablecoin_depeg_warn_bps:
            triggers.append(ShockTrigger(
                "DEPEG_20BPS", features.stablecoin_depeg_bps, self.config.stablecoin_depeg_warn_bps,
                None, timestamp_ms
            ))
        
        # Store triggers for audit trail
        self.recent_triggers.extend(triggers)
        
        # Determine severity level based on N-of-M logic
        trigger_count = len(triggers)
        
        # Check for halt-level triggers specifically
        halt_triggers = [t for t in triggers if any(x in t.type for x in ['_8SIG', 'VENUE_STALE', 'DEPEG_50BPS'])]
        if len(halt_triggers) >= self.config.min_triggers_halt or trigger_count >= self.config.min_triggers_halt + 2:
            return "halt", triggers
        
        # Check for flatten-level triggers
        flatten_triggers = [t for t in triggers if any(x in t.type for x in ['_6SIG', 'RV_4SIG', 'VPIN_98PCTL'])]
        if len(flatten_triggers) >= self.config.min_triggers_flatten or trigger_count >= self.config.min_triggers_flatten + 1:
            return "flatten", triggers
        
        # Check for derisk-level triggers
        derisk_triggers = [t for t in triggers if any(x in t.type for x in ['_5SIG', 'RV_3SIG', 'SPREAD_3SIG', 'VPIN_95PCTL', 'LAMBDA_3SIG'])]
        if len(derisk_triggers) >= self.config.min_triggers_derisk or trigger_count >= self.config.min_triggers_derisk:
            return "derisk", triggers
        
        # Check for warn-level triggers
        if trigger_count >= self.config.min_triggers_warn:
            return "warn", triggers
        
        return "normal", triggers
    
    def get_audit_summary(self, lookback_sec: float = 300.0) -> Dict[str, Any]:
        """Get audit summary of recent triggers and detections."""
        cutoff_ms = int(time.time() * 1000) - int(lookback_sec * 1000)
        recent = [t for t in self.recent_triggers if t.timestamp_ms >= cutoff_ms]
        
        trigger_counts = {}
        for trigger in recent:
            trigger_counts[trigger.type] = trigger_counts.get(trigger.type, 0) + 1
        
        return {
            "total_triggers": len(recent),
            "trigger_types": trigger_counts,
            "unique_trigger_types": len(trigger_counts),
            "lookback_sec": lookback_sec,
        }


__all__ = ["ShockDetector", "ShockConfig", "ShockFeatures", "ShockTrigger"]
