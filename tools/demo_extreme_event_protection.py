#!/usr/bin/env python3
"""
Example usage of Sprint 65 Extreme Event Protection system.

This script demonstrates how to integrate the extreme event protection
into a live trading system.
"""
import asyncio
import time
import random
from typing import Dict, Any

from ultra_signals.risk import (
    create_extreme_event_protection,
    ExtremeEventProtectionManager,
    ExtremeEventStatus
)
from ultra_signals.core.config import (
    ExtremeEventProtectionSettings,
    ShockDetectionSettings,
    CircuitBreakerPolicySettings
)


class MockOrderManager:
    """Mock order manager for demonstration."""
    
    def __init__(self):
        self.orders = []
    
    async def get_open_orders(self, symbol=None):
        return [{"id": f"order_{i}", "symbol": "BTCUSD"} for i in range(3)]
    
    async def cancel_order(self, order_id, symbol=None, reason=None):
        print(f"    💀 Cancelled order {order_id} (reason: {reason})")
        return {"success": True}
    
    async def place_order(self, **kwargs):
        print(f"    📝 Placed order: {kwargs}")
        return {"success": True, "slippage_bps": random.uniform(1, 3)}


class MockPositionManager:
    """Mock position manager for demonstration."""
    
    async def get_positions(self, symbols=None):
        return {
            "BTCUSD": {"quantity": 10.0, "mark_price": 50000},
            "ETHUSD": {"quantity": -5.0, "mark_price": 3000}
        }


class MockVenueRouter:
    """Mock venue router for demonstration."""
    
    def get_venue_health(self, venue):
        return random.uniform(0.7, 1.0)
    
    def get_available_venues(self):
        return ["binance", "okx", "bybit"]


class MockMarketData:
    """Mock market data provider."""
    
    async def get_recent_volume(self, symbol, window_sec):
        return {"avg_volume_per_sec": random.uniform(1000, 5000)}


def create_demo_settings() -> ExtremeEventProtectionSettings:
    """Create demo settings with relaxed thresholds for testing."""
    return ExtremeEventProtectionSettings(
        enabled=True,
        shock_detection=ShockDetectionSettings(
            warn_k_sigma=2.0,    # Lower thresholds for demo
            derisk_k_sigma=3.0,
            flatten_k_sigma=4.0,
            halt_k_sigma=5.0,
            min_triggers_warn=1,
            min_triggers_derisk=1,
            min_triggers_flatten=2
        ),
        circuit_policy=CircuitBreakerPolicySettings(
            warn_cooldown_bars=2,    # Shorter cooldowns for demo
            derisk_cooldown_bars=3,
            flatten_cooldown_bars=5,
            enable_staged_recovery=True,
            recovery_stages=[0.25, 0.5, 1.0],  # Faster recovery
            recovery_stage_bars=2
        )
    )


class TradingSimulator:
    """Simulates a trading system with extreme event protection."""
    
    def __init__(self):
        # Create settings
        self.settings = create_demo_settings()
        
        # Telegram settings (dry run mode)
        self.telegram_settings = {
            "telegram": {
                "enabled": True,
                "dry_run": True,
                "bot_token": "demo_token",
                "chat_id": "demo_chat"
            }
        }
        
        # Create protection manager
        self.protection = create_extreme_event_protection(
            settings=self.settings,
            symbol="BTCUSD",
            telegram_settings=self.telegram_settings
        )
        
        # Mock dependencies
        self.order_manager = MockOrderManager()
        self.position_manager = MockPositionManager()
        self.venue_router = MockVenueRouter()
        self.market_data = MockMarketData()
        
        # Inject dependencies
        self.protection.inject_execution_dependencies(
            order_manager=self.order_manager,
            position_manager=self.position_manager,
            venue_router=self.venue_router,
            market_data=self.market_data
        )
        
        # Register callbacks
        self.protection.register_callback("level_change", self.on_level_change)
        self.protection.register_callback("size_mult_change", self.on_size_change)
        self.protection.register_callback("cancel_orders", self.on_cancel_orders)
        self.protection.register_callback("flatten_positions", self.on_flatten_positions)
        
        # Simulation state
        self.current_price = 50000.0
        self.bar_count = 0
        self.base_position_size = 1000.0  # USD
        
    def on_level_change(self, params: Dict[str, Any]):
        """Handle circuit level changes."""
        old_level = params["old_level"]
        new_level = params["new_level"]
        triggers = params.get("triggers", [])
        
        print(f"\n🚨 CIRCUIT BREAKER: {old_level} → {new_level.upper()}")
        print(f"📊 Triggers: {[t.type for t in triggers]}")
        
        if new_level == "halt":
            print("🔴 TRADING HALTED - Extreme event detected!")
        elif new_level == "flatten":
            print("🛑 FLATTENING POSITIONS - Market stress detected")
        elif new_level == "derisk":
            print("🛡️ DE-RISK MODE - New entries blocked")
        elif new_level == "warn":
            print("⚠️ SHOCK WATCH - Size reduced")
        else:
            print("✅ NORMAL TRADING - Conditions normalized")
    
    def on_size_change(self, params: Dict[str, Any]):
        """Handle size multiplier changes."""
        size_mult = params["size_mult"]
        level = params["level"]
        print(f"📏 Size multiplier: {size_mult:.2f} (level: {level})")
    
    def on_cancel_orders(self, params: Dict[str, Any]):
        """Handle order cancellation requests."""
        reason = params.get("reason", "unknown")
        print(f"🔥 CANCELLING ALL ORDERS (reason: {reason})")
    
    def on_flatten_positions(self, params: Dict[str, Any]):
        """Handle position flattening requests."""
        style = params.get("style", "unknown")
        print(f"🏃 FLATTENING ALL POSITIONS (style: {style})")
    
    def simulate_tick(self, price_change_pct: float = None):
        """Simulate a price tick."""
        # Generate price movement
        if price_change_pct is not None:
            price_move = price_change_pct
        else:
            price_move = random.gauss(0, 0.001)  # 0.1% volatility
        
        self.current_price *= (1 + price_move)
        
        # Update market data
        self.protection.update_market_data(
            price=self.current_price,
            spread_bps=random.uniform(3, 8),
            top_bid_qty=random.uniform(500, 2000),
            top_ask_qty=random.uniform(500, 2000),
            ref_depth=1000,
            realized_vol=abs(price_move) * 100,  # Annualized %
            vpin=random.uniform(0.1, 0.3),
            vpin_pctl=random.uniform(0.1, 0.3),
            lambda_val=random.uniform(0, 0.001),
            venue_health=self.venue_router.get_venue_health("binance")
        )
        
        return price_move
    
    def simulate_bar_close(self) -> ExtremeEventStatus:
        """Simulate a bar close and update protection."""
        self.bar_count += 1
        
        # Update protection system
        status = self.protection.update(bar_close=True)
        
        # Print status
        print(f"\n📊 Bar {self.bar_count} | Price: ${self.current_price:,.0f} | Level: {status.circuit_level.upper()}")
        
        if status.circuit_level != "normal":
            print(f"   Threat Score: {status.threat_score:.2f}")
            print(f"   Size Mult: {status.size_mult_current:.2f}")
            if status.countdown_bars > 0:
                print(f"   Countdown: {status.countdown_bars} bars")
            if status.recovery_stage > 0:
                print(f"   Recovery: Stage {status.recovery_stage}")
        
        return status
    
    def calculate_position_size(self, signal_strength: float = 1.0) -> float:
        """Calculate position size with circuit breaker controls."""
        # Check if we can enter positions
        can_enter, reason = self.protection.can_enter_position()
        if not can_enter:
            print(f"   ❌ Position entry blocked: {reason}")
            return 0.0
        
        # Apply size multiplier
        size_mult = self.protection.get_effective_size_multiplier()
        effective_size = self.base_position_size * signal_strength * size_mult
        
        # Apply leverage cap if active
        leverage_cap = self.protection.get_effective_leverage_cap()
        if leverage_cap:
            max_size = 100000 / leverage_cap  # Assume $100k equity
            effective_size = min(effective_size, max_size)
            print(f"   🔒 Leverage cap: {leverage_cap:.1f}x")
        
        if effective_size < self.base_position_size:
            print(f"   📏 Size reduced: ${effective_size:.0f} (mult: {size_mult:.2f})")
        
        return effective_size
    
    async def run_simulation(self, bars: int = 50):
        """Run the complete simulation."""
        print("🚀 Starting Extreme Event Protection Demo")
        print("=" * 60)
        
        # Simulate normal trading
        for i in range(bars):
            # Normal price movement
            if i < 15:
                self.simulate_tick()
            
            # Inject shock event
            elif i == 15:
                print("\n💥 INJECTING SHOCK EVENT - 8% price drop!")
                self.simulate_tick(-0.08)  # 8% drop
            
            # Continue with elevated volatility
            elif i < 25:
                vol_mult = 3.0  # 3x normal volatility
                price_move = random.gauss(0, 0.001 * vol_mult)
                self.simulate_tick(price_move)
            
            # Return to normal
            else:
                self.simulate_tick()
            
            # Bar close
            status = self.simulate_bar_close()
            
            # Simulate trading decision
            if i % 5 == 0:  # Every 5 bars
                signal_strength = random.uniform(0.5, 1.5)
                size = self.calculate_position_size(signal_strength)
                if size > 0:
                    print(f"   📈 Would place order: ${size:.0f}")
            
            # Add delay for readability
            await asyncio.sleep(0.5)
        
        print("\n" + "=" * 60)
        print("📈 Simulation Complete")
        
        # Print final telemetry
        telemetry = self.protection.get_telemetry()
        circuit_stats = telemetry.get("circuit", {})
        
        print(f"\n📊 Final Statistics:")
        print(f"   Circuit Activations: {len(circuit_stats.get('recent_state_changes', []))}")
        print(f"   Current Level: {circuit_stats.get('circuit_state', 'unknown')}")
        print(f"   Final Size Mult: {circuit_stats.get('size_mult_current', 1.0):.2f}")
        
        shock_stats = telemetry.get("shock_detector", {})
        print(f"   Total Triggers: {shock_stats.get('total_triggers', 0)}")
        print(f"   Trigger Types: {shock_stats.get('trigger_types', {})}")


async def main():
    """Main demo function."""
    simulator = TradingSimulator()
    await simulator.run_simulation(bars=30)


if __name__ == "__main__":
    print("Sprint 65 - Extreme Event Protection Demo")
    print("This demo simulates a trading system with circuit breakers.\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
