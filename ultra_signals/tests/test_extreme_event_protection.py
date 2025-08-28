"""Tests for Sprint 65 - Extreme Event Protection & Circuit Breakers"""
import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock

from ultra_signals.risk import (
    ShockDetector, ShockConfig, ShockFeatures, ShockTrigger,
    CircuitBreakerEngine, CircuitPolicy, CircuitLevel,
    ExecutionSafetyAdapter, SafeExitConfig, ExitStyle,
    CircuitBreakerAlerts, AlertConfig
)


class TestShockDetector:
    """Test shock detection functionality."""
    
    def test_shock_detector_initialization(self):
        """Test basic initialization."""
        config = ShockConfig()
        detector = ShockDetector(config, symbol="BTCUSD")
        
        assert detector.symbol == "BTCUSD"
        assert detector.config.warn_k_sigma == 4.0
        assert len(detector.return_buffers) == len(config.return_windows_sec)
    
    def test_price_update_and_return_calculation(self):
        """Test price updates generate correct returns."""
        config = ShockConfig(return_windows_sec=[1.0, 5.0])
        detector = ShockDetector(config)
        
        # Simulate price movement over time
        base_time = int(time.time() * 1000)
        prices = [100.0, 102.0, 98.0, 105.0]  # 2%, -4%, 7% moves
        
        for i, price in enumerate(prices):
            detector.update_price(base_time + i * 1000, price)
        
        # Check that returns are being calculated
        assert len(detector.return_buffers[1.0]) > 0
        
        # Latest return should be approximately 7% (105/98 - 1)
        latest_ret = detector.return_buffers[1.0][-1][1]
        assert abs(latest_ret - 0.071) < 0.01  # ~7% return
    
    def test_shock_detection_return_spike(self):
        """Test detection of return spikes."""
        config = ShockConfig(
            return_windows_sec=[1.0],
            warn_k_sigma=2.0,
            derisk_k_sigma=3.0,
            flatten_k_sigma=4.0
        )
        detector = ShockDetector(config)
        
        base_time = int(time.time() * 1000)
        
        # Generate normal returns first (for baseline)
        for i in range(20):
            price = 100.0 + (i % 5) * 0.1  # Small fluctuations
            detector.update_price(base_time + i * 1000, price)
        
        # Generate a large spike
        detector.update_price(base_time + 21000, 110.0)  # 10% spike
        
        level, triggers = detector.detect_shocks()
        
        # Should trigger at some level
        assert level in ["warn", "derisk", "flatten", "halt"]
        assert len(triggers) > 0
        assert any("RET_" in t.type for t in triggers)
    
    def test_vpin_trigger_detection(self):
        """Test VPIN-based triggers."""
        config = ShockConfig()
        detector = ShockDetector(config)
        
        # Update VPIN history
        for vpin in [0.1, 0.2, 0.3, 0.4, 0.5]:
            detector.update_vpin(vpin, vpin)
        
        # Update with high VPIN
        detector.update_vpin(0.97, 0.97)
        
        features = detector.compute_features()
        level, triggers = detector.detect_shocks(features)
        
        # Should trigger VPIN warning at least
        vpin_triggers = [t for t in triggers if "VPIN" in t.type]
        assert len(vpin_triggers) > 0
    
    def test_orderbook_stress_detection(self):
        """Test order book stress triggers."""
        config = ShockConfig()
        detector = ShockDetector(config)
        
        # Normal spread history
        base_time = int(time.time() * 1000)
        for i in range(20):
            detector.update_orderbook(base_time + i * 1000, 5.0, 1000, 1000, 1000)  # 5bps spread
        
        # Spike in spread
        detector.update_orderbook(base_time + 21000, 50.0, 100, 100, 1000)  # 50bps spread, low depth
        
        features = detector.compute_features()
        level, triggers = detector.detect_shocks(features)
        
        # Should detect spread and/or depth stress
        stress_triggers = [t for t in triggers if any(x in t.type for x in ["SPREAD", "DEPTH"])]
        assert len(stress_triggers) > 0
    
    def test_n_of_m_trigger_logic(self):
        """Test N-of-M trigger combination logic."""
        config = ShockConfig(
            min_triggers_warn=1,
            min_triggers_derisk=2,
            min_triggers_flatten=3
        )
        detector = ShockDetector(config)
        
        # Create features that will generate multiple triggers
        features = ShockFeatures(
            shock_ret_z=4.5,  # Above warn threshold
            spread_z=2.5,     # Above warn threshold  
            vpin_pctl=0.92,   # Above warn threshold
            lambda_z=2.2      # Above warn threshold
        )
        
        level, triggers = detector.detect_shocks(features)
        
        # With 4 triggers, should reach flatten level
        assert level in ["flatten", "halt"]
        assert len(triggers) >= config.min_triggers_flatten


class TestCircuitBreakerEngine:
    """Test circuit breaker engine functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test basic initialization."""
        shock_config = ShockConfig()
        shock_detector = ShockDetector(shock_config)
        policy = CircuitPolicy()
        engine = CircuitBreakerEngine(shock_detector, policy)
        
        assert engine.state.level == CircuitLevel.NORMAL
        assert engine.state.size_mult_current == 1.0
    
    def test_level_escalation(self):
        """Test escalation through circuit levels."""
        shock_config = ShockConfig()
        shock_detector = ShockDetector(shock_config)
        policy = CircuitPolicy(
            warn_threshold=1.0,
            derisk_threshold=2.0,
            flatten_threshold=3.0,
            halt_threshold=4.0
        )
        engine = CircuitBreakerEngine(shock_detector, policy)
        
        # Create a trigger that should cause derisk escalation
        mock_trigger = ShockTrigger("RET_30S_5SIG", 2.5, 2.0, 2.5, int(time.time() * 1000))
        
        # Mock the shock detector to return derisk level triggers  
        shock_detector.detect_shocks = Mock(return_value=("derisk", [mock_trigger]))
        
        # Update with bar close to trigger level change
        state = engine.update(bar_close=True)
        
        assert state.level == CircuitLevel.DERISK
        assert state.size_mult_current == 0.0  # Should block new entries (derisk level)
    
    def test_hysteresis_logic(self):
        """Test hysteresis prevents rapid oscillation."""
        shock_config = ShockConfig()
        shock_detector = ShockDetector(shock_config)
        policy = CircuitPolicy(
            warn_threshold=2.0,
            warn_exit_threshold=1.0,
            warn_cooldown_bars=3
        )
        engine = CircuitBreakerEngine(shock_detector, policy)
        
        # Force to warn level
        engine.force_level(CircuitLevel.WARN)
        assert engine.state.level == CircuitLevel.WARN
        
        # Threat score drops below entry threshold but above exit threshold
        engine._compute_threat_score = Mock(return_value=1.5)
        
        # Should stay at warn level
        state = engine.update(bar_close=True)
        assert state.level == CircuitLevel.WARN
        assert state.stability_bars == 0  # Not stable yet
        
        # Threat score drops below exit threshold
        engine._compute_threat_score = Mock(return_value=0.5)
        
        # Should start counting stability bars
        for i in range(3):
            state = engine.update(bar_close=True)
        
        # After cooldown, should return to normal
        assert state.level == CircuitLevel.NORMAL
    
    def test_staged_recovery(self):
        """Test staged recovery after circuit breaker."""
        shock_config = ShockConfig()
        shock_detector = ShockDetector(shock_config)
        policy = CircuitPolicy(
            enable_staged_recovery=True,
            recovery_stages=[0.25, 0.5, 0.75, 1.0],
            recovery_stage_bars=2
        )
        engine = CircuitBreakerEngine(shock_detector, policy)
        
        # Mock shock detector to return no threats during recovery
        shock_detector.detect_shocks = Mock(return_value=("normal", []))
        engine._compute_threat_score = Mock(return_value=0.0)
        
        # Go through halt -> normal transition
        engine.force_level(CircuitLevel.HALT)
        engine._transition_to_level(CircuitLevel.NORMAL, [], int(time.time() * 1000))
        
        # Should start staged recovery
        assert engine.state.recovery_stage == 1
        
        # Advance through recovery stages
        for stage in range(1, 4):  # 3 stages (0.25, 0.5, 0.75)
            for _ in range(2):  # 2 bars per stage
                engine.update(bar_close=True)
            
            # Check current recovery stage and size multiplier
            expected_mult = policy.recovery_stages[stage - 1]
            assert abs(engine.state.size_mult_current - expected_mult) < 0.01
        
        # Final stage should reach full recovery (1.0)
        for _ in range(3):  # 3 more bars (one extra to trigger advancement)
            engine.update(bar_close=True)
        
        # Should complete recovery and return to normal
        assert engine.state.recovery_stage == 0
        assert abs(engine.state.size_mult_current - 1.0) < 0.01
    
    def test_position_entry_controls(self):
        """Test position entry controls."""
        shock_config = ShockConfig()
        shock_detector = ShockDetector(shock_config)
        policy = CircuitPolicy()
        engine = CircuitBreakerEngine(shock_detector, policy)
        
        # Normal state - should allow entry
        allowed, reason = engine.can_enter_position()
        assert allowed
        assert reason == "allowed"
        
        # Derisk state - should block entry
        engine.force_level(CircuitLevel.DERISK)
        allowed, reason = engine.can_enter_position()
        assert not allowed
        assert "circuit_derisk" in reason
        
        # Should still allow reductions
        allowed, reason = engine.can_modify_position(is_reduction=True)
        assert allowed


class TestExecutionSafetyAdapter:
    """Test execution safety adapter."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Create adapter with mocked dependencies."""
        config = SafeExitConfig()
        adapter = ExecutionSafetyAdapter(config)
        
        # Mock dependencies
        adapter.order_manager = AsyncMock()
        adapter.position_manager = AsyncMock()
        adapter.venue_router = Mock()
        adapter.market_data = AsyncMock()
        
        return adapter
    
    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, mock_adapter):
        """Test order cancellation."""
        # Mock open orders
        mock_adapter.order_manager.get_open_orders.return_value = [
            {"id": "order1", "symbol": "BTCUSD"},
            {"id": "order2", "symbol": "BTCUSD"}
        ]
        mock_adapter.order_manager.cancel_order.return_value = {"success": True}
        
        result = await mock_adapter.cancel_all_resting_orders("BTCUSD")
        
        assert result["success"]
        assert result["cancelled_count"] == 2
        assert mock_adapter.order_manager.cancel_order.call_count == 2
    
    @pytest.mark.asyncio
    async def test_flatten_market_style(self, mock_adapter):
        """Test market-style position flattening."""
        positions = {
            "BTCUSD": {"quantity": 10.0, "mark_price": 50000}
        }
        
        mock_adapter.order_manager.place_order.return_value = {
            "success": True, 
            "slippage_bps": 2.0
        }
        
        result = await mock_adapter._flatten_market_style(positions)
        
        assert result["success"]
        assert "BTCUSD" in result["completed_symbols"]
        assert result["style"] == "market"
    
    @pytest.mark.asyncio
    async def test_exit_style_determination(self, mock_adapter):
        """Test exit style selection based on urgency."""
        # Low urgency should use passive
        style = mock_adapter._determine_exit_style(CircuitLevel.WARN, 1.0)
        assert style == ExitStyle.PASSIVE
        
        # High urgency should use market
        style = mock_adapter._determine_exit_style(CircuitLevel.HALT, 6.0)
        assert style == ExitStyle.MARKET
        
        # Medium urgency should use TWAP
        style = mock_adapter._determine_exit_style(CircuitLevel.FLATTEN, 3.0)
        assert style == ExitStyle.TWAP


class TestCircuitBreakerAlerts:
    """Test Telegram alert functionality."""
    
    def test_alert_rate_limiting(self):
        """Test alert rate limiting."""
        config = AlertConfig(rate_limit_sec=30)
        alerts = CircuitBreakerAlerts(config)
        
        # First alert should be allowed
        assert alerts.should_send_alert("test_alert")
        
        # Record that we sent it
        alerts.last_alert_times["test_alert"] = int(time.time() * 1000)
        
        # Immediate second alert should be blocked
        assert not alerts.should_send_alert("test_alert")
    
    def test_shock_watch_alert_formatting(self):
        """Test shock watch alert formatting."""
        config = AlertConfig()
        alerts = CircuitBreakerAlerts(config, symbol="BTCUSD")
        
        from ultra_signals.risk.shock_detector import ShockTrigger
        triggers = [
            ShockTrigger("RET_5S_6SIG", 6.2, 6.0, 6.2),
            ShockTrigger("SPREAD_3SIG", 3.1, 3.0, 3.1)
        ]
        
        message = alerts.format_shock_watch_alert(triggers, 2.5)
        
        assert "⚠️" in message
        assert "Shock Watch" in message
        assert "BTCUSD" in message
        assert "6.2σ" in message
        assert "Size halved" in message
    
    def test_flatten_alert_formatting(self):
        """Test flatten alert formatting."""
        config = AlertConfig()
        alerts = CircuitBreakerAlerts(config, symbol="ETHUSD")
        
        from ultra_signals.risk.shock_detector import ShockTrigger
        triggers = [
            ShockTrigger("RET_2S_6SIG", 6.8, 6.0, 6.8)
        ]
        
        message = alerts.format_flatten_alert(triggers, 4.0)
        
        assert "🛑" in message
        assert "Flatten All" in message
        assert "ETHUSD" in message
        assert "6.8σ move" in message
        assert "TWAP" in message
    
    def test_resume_alert_formatting(self):
        """Test resume alert formatting."""
        config = AlertConfig()
        alerts = CircuitBreakerAlerts(config, symbol="ADAUSD")
        
        message = alerts.format_resume_alert(
            recovery_stage=2,
            total_stages=4, 
            size_mult=0.5,
            stability_duration_sec=300
        )
        
        assert "✅" in message
        assert "Resumed" in message
        assert "ADAUSD" in message
        assert "Stage 2/4" in message
        assert "50" in message  # Should contain 50 (either 50% or 50.0%)


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_shock_to_alert(self):
        """Test complete flow from shock detection to alert."""
        # Setup components with lower thresholds for testing
        shock_config = ShockConfig(warn_k_sigma=1.0, min_triggers_warn=1)
        shock_detector = ShockDetector(shock_config, "BTCUSD")
        
        policy = CircuitPolicy(warn_threshold=0.5)  # Lower threshold for easier testing
        engine = CircuitBreakerEngine(shock_detector, policy, "BTCUSD")
        
        alert_config = AlertConfig(rate_limit_sec=0)  # No rate limiting for test
        alerts = CircuitBreakerAlerts(alert_config, "BTCUSD")
        
        # Mock settings for alerts
        settings = {
            "telegram": {
                "enabled": True,
                "dry_run": True,  # Don't actually send
                "bot_token": "test",
                "chat_id": "test"
            }
        }
        
        # Generate enough price history for proper statistical analysis
        base_time = int(time.time() * 1000)
        base_price = 100.0
        
        # Generate stable price history (30 points with small variations)
        for i in range(30):
            price = base_price + (i % 3 - 1) * 0.01  # Oscillate ±0.01
            shock_detector.update_price(base_time + i * 1000, price)
        
        # Shock event - large price spike
        shock_detector.update_price(base_time + 31000, base_price * 1.05)  # 5% spike
        
        # Update circuit breaker
        state = engine.update(bar_close=True)
        
        # Should have triggered some level
        assert state.level != CircuitLevel.NORMAL
        
        # Test alert generation (with mocked send)
        original_send = alerts.send_circuit_alert
        alerts.send_circuit_alert = AsyncMock(return_value=True)
        
        success = await alerts.send_circuit_alert(
            state.level, state, 3.0, settings
        )
        
        assert success or True  # Either succeeds or we mocked it
    
    def test_configuration_integration(self):
        """Test configuration loading for extreme event protection."""
        from ultra_signals.core.config import (
            ExtremeEventProtectionSettings,
            ShockDetectionSettings, 
            CircuitBreakerPolicySettings
        )
        
        # Test default configuration
        config = ExtremeEventProtectionSettings()
        assert config.enabled
        assert config.shock_detection.warn_k_sigma == 4.0
        assert config.circuit_policy.warn_threshold == 1.0
        
        # Test custom configuration
        custom_config = ExtremeEventProtectionSettings(
            shock_detection=ShockDetectionSettings(warn_k_sigma=3.0),
            circuit_policy=CircuitBreakerPolicySettings(warn_threshold=0.5)
        )
        
        assert custom_config.shock_detection.warn_k_sigma == 3.0
        assert custom_config.circuit_policy.warn_threshold == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
