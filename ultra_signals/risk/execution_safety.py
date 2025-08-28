"""Extreme Event Protection - Execution Safety Adapter (Sprint 65)

Safe execution controls during circuit breaker states.
Handles graceful position flattening, order cancellation, and exit style preferences.
"""
from __future__ import annotations
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from .circuit_breaker_engine import CircuitLevel, CircuitState


class ExitStyle(Enum):
    """Exit execution styles."""
    PASSIVE = "passive"      # Post-only, market making style
    TWAP = "twap"           # Time-weighted average price
    SMART = "smart"         # Adaptive based on conditions
    MARKET = "market"       # Immediate market orders


@dataclass
class SafeExitConfig:
    """Configuration for safe position exits."""
    max_participation_rate: float = 0.1  # Max % of volume to use
    slice_duration_sec: int = 30  # Duration per TWAP slice
    max_slices: int = 10  # Max slices for TWAP
    passive_timeout_sec: int = 120  # Timeout before escalating from passive
    market_urgency_threshold: float = 5.0  # Threat score to force market orders
    allow_cross_spread: bool = False  # Allow crossing spread in passive mode
    min_order_value_usd: float = 10.0  # Min order size
    venue_health_threshold: float = 0.7  # Min venue health for safe exit


@dataclass
class ExitOrder:
    """Represents an exit order."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    style: ExitStyle
    urgency: float = 1.0
    venue_preference: Optional[str] = None
    max_slippage_bps: Optional[float] = None
    timeout_sec: Optional[int] = None


@dataclass
class FlattenerState:
    """State of position flattening process."""
    active: bool = False
    start_time_ms: int = 0
    positions_to_close: Dict[str, float] = None  # symbol -> quantity
    pending_orders: List[str] = None  # order IDs
    completed_quantity: float = 0.0
    target_quantity: float = 0.0
    style: ExitStyle = ExitStyle.PASSIVE
    participation_cap: float = 0.1
    escalation_count: int = 0
    last_update_ms: int = 0


class ExecutionSafetyAdapter:
    """Handles safe execution during extreme events.
    
    Provides methods for:
    - Graceful position flattening 
    - Emergency order cancellation
    - Exit style adaptation based on circuit level
    - Venue health consideration
    """
    
    def __init__(self, config: SafeExitConfig):
        self.config = config
        self.flattener_state = FlattenerState()
        
        # External interfaces (to be injected)
        self.order_manager = None  # For placing/cancelling orders
        self.position_manager = None  # For getting current positions
        self.venue_router = None  # For venue health and routing
        self.market_data = None  # For price/volume data
        
        # Metrics
        self.exit_metrics = {
            "total_exits_attempted": 0,
            "successful_exits": 0,
            "failed_exits": 0,
            "avg_slippage_bps": 0.0,
            "avg_completion_time_sec": 0.0
        }
    
    def inject_dependencies(self, 
                           order_manager=None,
                           position_manager=None, 
                           venue_router=None,
                           market_data=None) -> None:
        """Inject external dependencies."""
        if order_manager:
            self.order_manager = order_manager
        if position_manager:
            self.position_manager = position_manager
        if venue_router:
            self.venue_router = venue_router
        if market_data:
            self.market_data = market_data
    
    async def cancel_all_resting_orders(self, symbol: Optional[str] = None,
                                       reason: str = "circuit_breaker") -> Dict[str, Any]:
        """Cancel all resting orders, optionally for specific symbol."""
        if not self.order_manager:
            logger.warning("[SafeExit] No order manager available for cancellation")
            return {"success": False, "reason": "no_order_manager"}
        
        try:
            # Get open orders
            open_orders = await self.order_manager.get_open_orders(symbol)
            
            if not open_orders:
                return {"success": True, "cancelled_count": 0}
            
            # Cancel orders in batches to avoid overwhelming the exchange
            batch_size = 10
            cancelled_count = 0
            failed_cancellations = []
            
            for i in range(0, len(open_orders), batch_size):
                batch = open_orders[i:i + batch_size]
                cancel_tasks = []
                
                for order in batch:
                    task = self.order_manager.cancel_order(
                        order_id=order.get("id"),
                        symbol=order.get("symbol"),
                        reason=reason
                    )
                    cancel_tasks.append(task)
                
                # Execute batch
                results = await asyncio.gather(*cancel_tasks, return_exceptions=True)
                
                for j, result in enumerate(results):
                    if isinstance(result, Exception):
                        failed_cancellations.append({
                            "order_id": batch[j].get("id"),
                            "error": str(result)
                        })
                    else:
                        cancelled_count += 1
                
                # Small delay between batches
                if i + batch_size < len(open_orders):
                    await asyncio.sleep(0.1)
            
            logger.info(f"[SafeExit] Cancelled {cancelled_count}/{len(open_orders)} orders | Reason: {reason}")
            
            return {
                "success": True,
                "cancelled_count": cancelled_count,
                "failed_count": len(failed_cancellations),
                "failures": failed_cancellations
            }
            
        except Exception as e:
            logger.error(f"[SafeExit] Error cancelling orders: {e}")
            return {"success": False, "error": str(e)}
    
    async def flatten_all_positions(self, 
                                   circuit_level: CircuitLevel,
                                   threat_score: float = 1.0,
                                   symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Flatten all positions using appropriate exit style."""
        if not self.position_manager:
            logger.warning("[SafeExit] No position manager available for flattening")
            return {"success": False, "reason": "no_position_manager"}
        
        # Get current positions
        try:
            positions = await self.position_manager.get_positions(symbols)
            
            if not positions:
                return {"success": True, "reason": "no_positions"}
            
            # Filter out zero/negligible positions
            significant_positions = {
                symbol: pos for symbol, pos in positions.items()
                if abs(pos.get("quantity", 0)) * pos.get("mark_price", 0) >= self.config.min_order_value_usd
            }
            
            if not significant_positions:
                return {"success": True, "reason": "no_significant_positions"}
            
            # Determine exit style based on circuit level and threat score
            exit_style = self._determine_exit_style(circuit_level, threat_score)
            participation_cap = self._determine_participation_cap(circuit_level)
            
            # Initialize flattener state
            self.flattener_state = FlattenerState(
                active=True,
                start_time_ms=int(time.time() * 1000),
                positions_to_close=significant_positions.copy(),
                pending_orders=[],
                target_quantity=sum(abs(pos.get("quantity", 0)) for pos in significant_positions.values()),
                style=exit_style,
                participation_cap=participation_cap,
                last_update_ms=int(time.time() * 1000)
            )
            
            logger.info(f"[SafeExit] Starting flatten: {len(significant_positions)} positions | Style: {exit_style.value}")
            
            # Execute flattening based on style
            if exit_style == ExitStyle.MARKET:
                result = await self._flatten_market_style(significant_positions)
            elif exit_style == ExitStyle.TWAP:
                result = await self._flatten_twap_style(significant_positions)
            elif exit_style == ExitStyle.SMART:
                result = await self._flatten_smart_style(significant_positions, threat_score)
            else:  # PASSIVE
                result = await self._flatten_passive_style(significant_positions)
            
            self.flattener_state.active = False
            self.exit_metrics["total_exits_attempted"] += 1
            
            if result.get("success"):
                self.exit_metrics["successful_exits"] += 1
            else:
                self.exit_metrics["failed_exits"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"[SafeExit] Error flattening positions: {e}")
            self.flattener_state.active = False
            return {"success": False, "error": str(e)}
    
    def _determine_exit_style(self, circuit_level: CircuitLevel, 
                            threat_score: float) -> ExitStyle:
        """Determine appropriate exit style based on urgency."""
        if threat_score >= self.config.market_urgency_threshold:
            return ExitStyle.MARKET
        elif circuit_level == CircuitLevel.HALT:
            return ExitStyle.MARKET
        elif circuit_level == CircuitLevel.FLATTEN:
            return ExitStyle.TWAP
        elif circuit_level == CircuitLevel.DERISK:
            return ExitStyle.SMART
        else:
            return ExitStyle.PASSIVE
    
    def _determine_participation_cap(self, circuit_level: CircuitLevel) -> float:
        """Determine participation cap based on circuit level."""
        if circuit_level == CircuitLevel.HALT:
            return 0.2  # Allow higher participation in emergency
        elif circuit_level == CircuitLevel.FLATTEN:
            return 0.1  # Standard TWAP participation
        elif circuit_level == CircuitLevel.DERISK:
            return 0.05  # Conservative participation
        else:
            return 0.03  # Very conservative
    
    async def _flatten_market_style(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten using immediate market orders."""
        if not self.order_manager:
            return {"success": False, "reason": "no_order_manager"}
        
        completed_symbols = []
        failed_symbols = []
        total_slippage_bps = 0.0
        
        for symbol, position in positions.items():
            try:
                quantity = abs(position.get("quantity", 0))
                side = "sell" if position.get("quantity", 0) > 0 else "buy"
                
                # Place market order
                order_result = await self.order_manager.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type="market",
                    reason="circuit_flatten_market"
                )
                
                if order_result.get("success"):
                    completed_symbols.append(symbol)
                    # Track slippage if available
                    slippage = order_result.get("slippage_bps", 0)
                    total_slippage_bps += slippage
                else:
                    failed_symbols.append(symbol)
                
                # Small delay between orders
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.error(f"[SafeExit] Market order failed for {symbol}: {e}")
                failed_symbols.append(symbol)
        
        avg_slippage = total_slippage_bps / max(len(completed_symbols), 1)
        
        return {
            "success": len(failed_symbols) == 0,
            "style": "market",
            "completed_symbols": completed_symbols,
            "failed_symbols": failed_symbols,
            "avg_slippage_bps": avg_slippage,
            "completion_time_sec": (int(time.time() * 1000) - self.flattener_state.start_time_ms) / 1000.0
        }
    
    async def _flatten_twap_style(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten using TWAP strategy."""
        if not self.order_manager or not self.market_data:
            return {"success": False, "reason": "missing_dependencies"}
        
        # Calculate TWAP parameters
        slice_duration = self.config.slice_duration_sec
        max_slices = self.config.max_slices
        participation_rate = self.flattener_state.participation_cap
        
        completed_symbols = []
        failed_symbols = []
        
        for symbol, position in positions.items():
            try:
                quantity = abs(position.get("quantity", 0))
                side = "sell" if position.get("quantity", 0) > 0 else "buy"
                
                # Get recent volume for participation calculation
                volume_data = await self.market_data.get_recent_volume(symbol, window_sec=300)
                avg_volume_per_sec = volume_data.get("avg_volume_per_sec", 1000)
                
                # Calculate slice size
                max_slice_by_participation = avg_volume_per_sec * slice_duration * participation_rate
                slice_size = min(quantity / max_slices, max_slice_by_participation)
                slice_size = max(slice_size, self.config.min_order_value_usd / position.get("mark_price", 1))
                
                remaining_qty = quantity
                slice_count = 0
                
                while remaining_qty > 0 and slice_count < max_slices:
                    current_slice = min(slice_size, remaining_qty)
                    
                    # Place limit order near mid (TWAP style)
                    order_result = await self.order_manager.place_twap_slice(
                        symbol=symbol,
                        side=side,
                        quantity=current_slice,
                        duration_sec=slice_duration,
                        participation_rate=participation_rate
                    )
                    
                    if order_result.get("success"):
                        remaining_qty -= current_slice
                        slice_count += 1
                        await asyncio.sleep(slice_duration)
                    else:
                        logger.warning(f"[SafeExit] TWAP slice failed for {symbol}: {order_result.get('error')}")
                        break
                
                if remaining_qty <= quantity * 0.1:  # 90% completion threshold
                    completed_symbols.append(symbol)
                else:
                    failed_symbols.append(symbol)
                
            except Exception as e:
                logger.error(f"[SafeExit] TWAP execution failed for {symbol}: {e}")
                failed_symbols.append(symbol)
        
        return {
            "success": len(failed_symbols) == 0,
            "style": "twap",
            "completed_symbols": completed_symbols,
            "failed_symbols": failed_symbols,
            "completion_time_sec": (int(time.time() * 1000) - self.flattener_state.start_time_ms) / 1000.0
        }
    
    async def _flatten_smart_style(self, positions: Dict[str, Any], 
                                  threat_score: float) -> Dict[str, Any]:
        """Flatten using adaptive smart execution."""
        # Start with passive, escalate based on conditions
        
        # Try passive first
        result = await self._flatten_passive_style(positions, timeout_sec=60)
        
        if result.get("success"):
            return result
        
        # Escalate to TWAP if passive fails
        remaining_positions = {
            symbol: pos for symbol, pos in positions.items()
            if symbol not in result.get("completed_symbols", [])
        }
        
        if remaining_positions:
            logger.info("[SafeExit] Escalating to TWAP for remaining positions")
            twap_result = await self._flatten_twap_style(remaining_positions)
            
            # Merge results
            return {
                "success": twap_result.get("success", False),
                "style": "smart",
                "completed_symbols": result.get("completed_symbols", []) + twap_result.get("completed_symbols", []),
                "failed_symbols": twap_result.get("failed_symbols", []),
                "escalation_used": True
            }
        
        return result
    
    async def _flatten_passive_style(self, positions: Dict[str, Any],
                                    timeout_sec: Optional[int] = None) -> Dict[str, Any]:
        """Flatten using passive post-only orders."""
        if not self.order_manager:
            return {"success": False, "reason": "no_order_manager"}
        
        timeout = timeout_sec or self.config.passive_timeout_sec
        start_time = time.time()
        
        completed_symbols = []
        failed_symbols = []
        
        for symbol, position in positions.items():
            try:
                quantity = abs(position.get("quantity", 0))
                side = "sell" if position.get("quantity", 0) > 0 else "buy"
                
                # Place post-only order at/near best price
                order_result = await self.order_manager.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type="limit_post_only",
                    reason="circuit_flatten_passive"
                )
                
                if order_result.get("success"):
                    order_id = order_result.get("order_id")
                    
                    # Wait for fill with timeout
                    filled = await self._wait_for_fill(order_id, timeout - (time.time() - start_time))
                    
                    if filled:
                        completed_symbols.append(symbol)
                    else:
                        # Cancel unfilled order
                        await self.order_manager.cancel_order(order_id)
                        failed_symbols.append(symbol)
                else:
                    failed_symbols.append(symbol)
                
            except Exception as e:
                logger.error(f"[SafeExit] Passive order failed for {symbol}: {e}")
                failed_symbols.append(symbol)
        
        return {
            "success": len(failed_symbols) == 0,
            "style": "passive",
            "completed_symbols": completed_symbols,
            "failed_symbols": failed_symbols,
            "timeout_reached": (time.time() - start_time) >= timeout
        }
    
    async def _wait_for_fill(self, order_id: str, timeout_sec: float) -> bool:
        """Wait for order to fill within timeout."""
        start_time = time.time()
        check_interval = min(1.0, timeout_sec / 10)  # Check 10 times during timeout
        
        while time.time() - start_time < timeout_sec:
            try:
                order_status = await self.order_manager.get_order_status(order_id)
                if order_status.get("status") == "filled":
                    return True
                elif order_status.get("status") in ["cancelled", "rejected"]:
                    return False
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"[SafeExit] Error checking order status {order_id}: {e}")
                return False
        
        return False
    
    def get_flattener_status(self) -> Dict[str, Any]:
        """Get current flattener status."""
        if not self.flattener_state.active:
            return {"active": False}
        
        elapsed_sec = (int(time.time() * 1000) - self.flattener_state.start_time_ms) / 1000.0
        completion_pct = (self.flattener_state.completed_quantity / max(self.flattener_state.target_quantity, 1)) * 100
        
        return {
            "active": True,
            "elapsed_sec": elapsed_sec,
            "completion_pct": completion_pct,
            "style": self.flattener_state.style.value,
            "positions_remaining": len(self.flattener_state.positions_to_close or {}),
            "pending_orders": len(self.flattener_state.pending_orders or []),
            "escalation_count": self.flattener_state.escalation_count
        }
    
    def get_exit_metrics(self) -> Dict[str, Any]:
        """Get exit execution metrics."""
        return dict(self.exit_metrics)
    
    def should_use_venue(self, venue: str, circuit_level: CircuitLevel) -> bool:
        """Check if venue is safe to use during circuit breaker."""
        if not self.venue_router:
            return True  # Default to allowing if no router
        
        venue_health = self.venue_router.get_venue_health(venue)
        
        # Higher standards during circuit breaker events
        if circuit_level in [CircuitLevel.FLATTEN, CircuitLevel.HALT]:
            return venue_health >= self.config.venue_health_threshold
        else:
            return venue_health >= 0.5  # Lower threshold for normal operations
    
    def get_preferred_venues(self, circuit_level: CircuitLevel) -> List[str]:
        """Get preferred venues for execution during circuit breaker."""
        if not self.venue_router:
            return []
        
        all_venues = self.venue_router.get_available_venues()
        safe_venues = [v for v in all_venues if self.should_use_venue(v, circuit_level)]
        
        # Sort by health score
        safe_venues.sort(key=lambda v: self.venue_router.get_venue_health(v), reverse=True)
        
        return safe_venues


__all__ = ["ExecutionSafetyAdapter", "SafeExitConfig", "ExitStyle", "ExitOrder", "FlattenerState"]
