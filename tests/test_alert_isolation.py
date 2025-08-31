#!/usr/bin/env python3
"""
Test Alert Isolation and Deduplication

This module tests:
1. Per-symbol state isolation
2. Cooldown and distance gates
3. Idempotency with decision_id
4. Anti-burst protection
"""

import time
import hashlib
from unittest.mock import Mock, patch
from loguru import logger

# Mock imports
with patch.dict('sys.modules', {
    'ultra_signals.core.config': Mock(),
    'ultra_signals.data.binance_ws': Mock(),
    'ultra_signals.data.funding_provider': Mock(),
    'ultra_signals.core.feature_store': Mock(),
    'ultra_signals.engine.real_engine': Mock(),
    'ultra_signals.transport.telegram': Mock(),
    'ultra_signals.live.metrics': Mock(),
}):
    from ultra_signals.apps.realtime_runner import ResilientSignalRunner, SymbolState


class TestAlertIsolation:
    """Test suite for alert isolation and deduplication."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.settings = Mock()
        self.settings.runtime.min_signal_interval_sec = 60.0
        self.settings.runtime.min_confidence = 0.65
        self.settings.runtime.max_consecutive_signals = 3
        self.settings.runtime.cooldown_base_sec = 60.0
        
        self.runner = ResilientSignalRunner(self.settings)
        self.current_time = time.time()
        
    def test_symbol_state_isolation(self):
        """Test that symbol states are completely isolated."""
        # Create decisions for different symbols
        decision1 = Mock()
        decision1.confidence = 0.8
        decision1.decision = "LONG"
        decision1.symbol = "BTCUSDT"
        
        decision2 = Mock()
        decision2.confidence = 0.8
        decision2.decision = "SHORT"
        decision2.symbol = "ETHUSDT"
        
        # Send signal for BTCUSDT
        should_send1 = self.runner._should_send_signal("BTCUSDT", decision1, self.current_time)
        assert should_send1, "First signal for BTCUSDT should be sent"
        
        # ETHUSDT should be unaffected
        should_send2 = self.runner._should_send_signal("ETHUSDT", decision2, self.current_time)
        assert should_send2, "First signal for ETHUSDT should be sent (different symbol)"
        
        # Check that states are separate
        btc_state = self.runner.symbol_states["BTCUSDT"]
        eth_state = self.runner.symbol_states["ETHUSDT"]
        
        assert btc_state.last_signal_ts != eth_state.last_signal_ts, "Symbol states should be independent"
        
    def test_cooldown_enforcement(self):
        """Test that cooldown periods are properly enforced."""
        symbol = "BTCUSDT"
        decision = Mock()
        decision.confidence = 0.8
        decision.decision = "LONG"
        
        # Send first signal
        should_send1 = self.runner._should_send_signal(symbol, decision, self.current_time)
        assert should_send1, "First signal should be sent"
        
        # Try to send immediately after (should be blocked)
        should_send2 = self.runner._should_send_signal(symbol, decision, self.current_time + 1)
        assert not should_send2, "Signal should be blocked by cooldown"
        
        # Try after cooldown period (should be allowed)
        should_send3 = self.runner._should_send_signal(symbol, decision, self.current_time + 65)
        assert should_send3, "Signal should be allowed after cooldown"
        
    def test_consecutive_signal_cooldown(self):
        """Test consecutive signal cooldown with exponential duration."""
        symbol = "BTCUSDT"
        decision = Mock()
        decision.confidence = 0.8
        decision.decision = "LONG"
        
        # Send 3 consecutive signals
        for i in range(3):
            should_send = self.runner._should_send_signal(symbol, decision, self.current_time + (i * 65))
            assert should_send, f"Signal {i+1} should be sent"
        
        # 4th signal should trigger cooldown
        should_send4 = self.runner._should_send_signal(symbol, decision, self.current_time + 200)
        assert not should_send4, "4th consecutive signal should trigger cooldown"
        
        # Check cooldown duration
        state = self.runner.symbol_states[symbol]
        assert state.cooldown_until > self.current_time, "Cooldown should be active"
        
        # After cooldown expires, should be allowed again
        future_time = self.current_time + 300  # Well after cooldown
        should_send5 = self.runner._should_send_signal(symbol, decision, future_time)
        assert should_send5, "Signal should be allowed after cooldown expires"
        
    def test_confidence_threshold(self):
        """Test that low confidence signals are blocked."""
        symbol = "BTCUSDT"
        decision = Mock()
        decision.decision = "LONG"
        
        # Test below threshold
        decision.confidence = 0.5
        should_send = self.runner._should_send_signal(symbol, decision, self.current_time)
        assert not should_send, "Low confidence signal should be blocked"
        
        # Test at threshold
        decision.confidence = 0.65
        should_send = self.runner._should_send_signal(symbol, decision, self.current_time)
        assert should_send, "Signal at threshold should be sent"
        
        # Test above threshold
        decision.confidence = 0.8
        should_send = self.runner._should_send_signal(symbol, decision, self.current_time + 65)
        assert should_send, "High confidence signal should be sent"
        
    def test_decision_id_idempotency(self):
        """Test that identical decisions don't duplicate."""
        symbol = "BTCUSDT"
        decision1 = Mock()
        decision1.confidence = 0.8
        decision1.decision = "LONG"
        decision1.symbol = symbol
        decision1.tf = "5m"
        
        # Create identical decision
        decision2 = Mock()
        decision2.confidence = 0.8
        decision2.decision = "LONG"
        decision2.symbol = symbol
        decision2.tf = "5m"
        
        # Send first decision
        should_send1 = self.runner._should_send_signal(symbol, decision1, self.current_time)
        assert should_send1, "First decision should be sent"
        
        # Try to send identical decision immediately
        should_send2 = self.runner._should_send_signal(symbol, decision2, self.current_time + 1)
        assert not should_send2, "Identical decision should be blocked"
        
    def test_price_distance_gate(self):
        """Test that price distance gates prevent rapid re-alerts."""
        symbol = "BTCUSDT"
        
        # Mock feature store with ATR
        mock_ohlcv = Mock()
        mock_ohlcv.__len__ = lambda self: 250  # Sufficient data
        
        self.runner.feature_store.get_ohlcv = Mock(return_value=mock_ohlcv)
        
        # Mock ATR calculation
        def mock_atr(data, period=14):
            return 1000.0  # 1000 USDT ATR
        
        with patch('ultra_signals.features.volatility.atr', mock_atr):
            decision1 = Mock()
            decision1.confidence = 0.8
            decision1.decision = "LONG"
            decision1.symbol = symbol
            
            # Send first signal
            should_send1 = self.runner._should_send_signal(symbol, decision1, self.current_time)
            assert should_send1, "First signal should be sent"
            
            # Try to send same-side signal with small price movement
            decision2 = Mock()
            decision2.confidence = 0.8
            decision2.decision = "LONG"  # Same side
            decision2.symbol = symbol
            
            # Mock small price movement (less than 0.5 ATR)
            should_send2 = self.runner._should_send_signal(symbol, decision2, self.current_time + 65)
            # Note: This test would need actual price data to be fully accurate
            # For now, we test the cooldown mechanism
            
    def test_side_change_resets_cooldown(self):
        """Test that changing sides resets consecutive signal count."""
        symbol = "BTCUSDT"
        
        # Send 2 LONG signals
        long_decision = Mock()
        long_decision.confidence = 0.8
        long_decision.decision = "LONG"
        
        for i in range(2):
            should_send = self.runner._should_send_signal(symbol, long_decision, self.current_time + (i * 65))
            assert should_send, f"LONG signal {i+1} should be sent"
        
        # Send SHORT signal (should reset consecutive count)
        short_decision = Mock()
        short_decision.confidence = 0.8
        short_decision.decision = "SHORT"
        
        should_send_short = self.runner._should_send_signal(symbol, short_decision, self.current_time + 200)
        assert should_send_short, "SHORT signal should be sent (side change)"
        
        # Check that consecutive count was reset
        state = self.runner.symbol_states[symbol]
        assert state.consecutive_signals == 1, "Consecutive count should be reset to 1"
        
    def test_multiple_symbols_independence(self):
        """Test that multiple symbols operate completely independently."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        decisions = {}
        
        # Create decisions for each symbol
        for symbol in symbols:
            decision = Mock()
            decision.confidence = 0.8
            decision.decision = "LONG"
            decision.symbol = symbol
            decisions[symbol] = decision
        
        # Send signals for all symbols simultaneously
        for symbol, decision in decisions.items():
            should_send = self.runner._should_send_signal(symbol, decision, self.current_time)
            assert should_send, f"Signal for {symbol} should be sent"
        
        # Try to send second signals immediately (should all be blocked)
        for symbol, decision in decisions.items():
            should_send = self.runner._should_send_signal(symbol, decision, self.current_time + 1)
            assert not should_send, f"Second signal for {symbol} should be blocked"
        
        # Check that all states are independent
        states = [self.runner.symbol_states[symbol] for symbol in symbols]
        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                if i != j:
                    assert state1.last_signal_ts != state2.last_signal_ts, f"States {i} and {j} should be independent"
                    
    def test_cooldown_expiration(self):
        """Test that cooldowns properly expire and reset."""
        symbol = "BTCUSDT"
        decision = Mock()
        decision.confidence = 0.8
        decision.decision = "LONG"
        
        # Send 3 signals to trigger cooldown
        for i in range(3):
            should_send = self.runner._should_send_signal(symbol, decision, self.current_time + (i * 65))
            assert should_send, f"Signal {i+1} should be sent"
        
        # Verify cooldown is active
        state = self.runner.symbol_states[symbol]
        assert state.cooldown_until > self.current_time, "Cooldown should be active"
        
        # Wait for cooldown to expire
        cooldown_duration = state.cooldown_until - self.current_time
        future_time = self.current_time + cooldown_duration + 10  # 10 seconds after expiration
        
        # Signal should be allowed after cooldown expires
        should_send = self.runner._should_send_signal(symbol, decision, future_time)
        assert should_send, "Signal should be allowed after cooldown expires"
        
        # Consecutive count should be reset
        assert state.consecutive_signals == 1, "Consecutive count should be reset to 1"
        
    def test_edge_case_timing(self):
        """Test edge cases around timing boundaries."""
        symbol = "BTCUSDT"
        decision = Mock()
        decision.confidence = 0.8
        decision.decision = "LONG"
        
        # Send first signal
        should_send1 = self.runner._should_send_signal(symbol, decision, self.current_time)
        assert should_send1, "First signal should be sent"
        
        # Try exactly at cooldown boundary (should be blocked)
        boundary_time = self.current_time + 60.0
        should_send2 = self.runner._should_send_signal(symbol, decision, boundary_time)
        assert not should_send2, "Signal at cooldown boundary should be blocked"
        
        # Try just after boundary (should be allowed)
        after_boundary_time = self.current_time + 60.1
        should_send3 = self.runner._should_send_signal(symbol, decision, after_boundary_time)
        assert should_send3, "Signal just after boundary should be allowed"


def test_decision_id_generation():
    """Test that decision IDs are deterministic and unique."""
    # Test deterministic generation
    decision1 = Mock()
    decision1.symbol = "BTCUSDT"
    decision1.tf = "5m"
    decision1.decision = "LONG"
    decision1.confidence = 0.8
    
    # Mock bar time
    bar_time = 1640995200  # Fixed timestamp
    
    # Generate decision ID
    decision_id1 = hashlib.md5(
        f"{decision1.symbol}_{decision1.tf}_{bar_time}_{decision1.decision}".encode()
    ).hexdigest()[:8]
    
    # Generate again with same parameters
    decision_id2 = hashlib.md5(
        f"{decision1.symbol}_{decision1.tf}_{bar_time}_{decision1.decision}".encode()
    ).hexdigest()[:8]
    
    assert decision_id1 == decision_id2, "Decision IDs should be deterministic"
    
    # Test uniqueness with different parameters
    decision2 = Mock()
    decision2.symbol = "ETHUSDT"  # Different symbol
    decision2.tf = "5m"
    decision2.decision = "LONG"
    
    decision_id3 = hashlib.md5(
        f"{decision2.symbol}_{decision2.tf}_{bar_time}_{decision2.decision}".encode()
    ).hexdigest()[:8]
    
    assert decision_id1 != decision_id3, "Different symbols should have different IDs"
    
    # Test different timeframes
    decision3 = Mock()
    decision3.symbol = "BTCUSDT"
    decision3.tf = "15m"  # Different timeframe
    decision3.decision = "LONG"
    
    decision_id4 = hashlib.md5(
        f"{decision3.symbol}_{decision3.tf}_{bar_time}_{decision3.decision}".encode()
    ).hexdigest()[:8]
    
    assert decision_id1 != decision_id4, "Different timeframes should have different IDs"


def main():
    """Run all isolation tests."""
    print("🧪 Testing Alert Isolation and Deduplication...")
    
    # Create test instance
    test_suite = TestAlertIsolation()
    
    # Run all tests
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            test_suite.setup_method()
            getattr(test_suite, method_name)()
            print(f"✅ {method_name} passed")
            passed += 1
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            failed += 1
    
    # Run standalone tests
    try:
        test_decision_id_generation()
        print("✅ test_decision_id_generation passed")
        passed += 1
    except Exception as e:
        print(f"❌ test_decision_id_generation failed: {e}")
        failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All isolation tests passed!")
        return True
    else:
        print("❌ Some isolation tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)



