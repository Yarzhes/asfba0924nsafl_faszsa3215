"""
Unit tests for signal gate NMS functionality.
"""

import pytest
import time
from unittest.mock import Mock

from ultra_signals.engine.signal_gate import (
    SignalGate, SignalRecord, create_signal_gate, apply_signal_gate
)


class TestSignalGate:
    """Test signal gate functionality."""
    
    def test_signal_gate_initialization(self):
        """Test signal gate initialization."""
        settings = {
            'gates': {
                'nms_window_bars': 5,
                'min_flip_distance_atr': 0.8,
                'cooldown_seconds': 300
            }
        }
        
        gate = SignalGate(settings)
        
        assert gate.nms_window_bars == 5
        assert gate.min_flip_distance_atr == 0.8
        assert gate.cooldown_seconds == 300
        assert gate.signal_history == {}
        assert gate.last_signal_time == {}
        assert gate.last_decision == {}
        assert gate.last_price == {}
    
    def test_signal_gate_default_settings(self):
        """Test signal gate with default settings."""
        settings = {}
        
        gate = SignalGate(settings)
        
        assert gate.nms_window_bars == 3
        assert gate.min_flip_distance_atr == 0.6
        assert gate.cooldown_seconds == 180
    
    def test_should_allow_signal_first_signal(self):
        """Test allowing the first signal for a symbol."""
        gate = SignalGate({})
        
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.8,
            price=50000.0,
            atr=1000.0
        )
        
        assert allowed is True
        assert reason == "allowed"
        assert "BTCUSDT" in gate.signal_history
        assert len(gate.signal_history["BTCUSDT"]) == 1
    
    def test_should_allow_signal_cooldown(self):
        """Test cooldown functionality."""
        gate = SignalGate({'gates': {'cooldown_seconds': 60}})
        
        # First signal
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.8,
            price=50000.0,
            atr=1000.0,
            current_time=1000.0
        )
        assert allowed is True
        
        # Second signal within cooldown
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="SHORT",
            confidence=0.9,
            price=50100.0,
            atr=1000.0,
            current_time=1030.0  # 30 seconds later
        )
        assert allowed is False
        assert reason == "cooldown"
    
    def test_should_allow_signal_nms_suppression(self):
        """Test non-maximum suppression."""
        gate = SignalGate({'gates': {'nms_window_bars': 3, 'cooldown_seconds': 1}})
        
        # First signal
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.7,
            price=50000.0,
            atr=1000.0,
            current_time=1000.0
        )
        assert allowed is True
        
        # Second signal of same side with lower confidence
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.6,
            price=50100.0,
            atr=1000.0,
            current_time=1001.0
        )
        assert allowed is False
        assert reason == "nms_suppressed"
        
        # Third signal of same side with higher confidence
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.9,
            price=50200.0,
            atr=1000.0,
            current_time=1002.0
        )
        assert allowed is True
        assert reason == "allowed"
    
    def test_should_allow_signal_flip_flop_guard(self):
        """Test flip-flop guard functionality."""
        gate = SignalGate({'gates': {'min_flip_distance_atr': 0.5, 'cooldown_seconds': 1}})
        
        # First signal
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.8,
            price=50000.0,
            atr=1000.0,
            current_time=1000.0
        )
        assert allowed is True
        
        # Second signal opposite direction with insufficient distance
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="SHORT",
            confidence=0.8,
            price=50020.0,  # Only 0.02 ATR distance
            atr=1000.0,
            current_time=1001.0
        )
        assert allowed is False
        assert reason == "flip_flop_guard"
        
        # Third signal opposite direction with sufficient distance
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="SHORT",
            confidence=0.8,
            price=50600.0,  # 0.6 ATR distance
            atr=1000.0,
            current_time=1002.0
        )
        assert allowed is True
        assert reason == "allowed"
    
    def test_should_allow_signal_different_sides(self):
        """Test that different sides don't trigger NMS."""
        gate = SignalGate({'gates': {'nms_window_bars': 3, 'cooldown_seconds': 1}})
        
        # First signal
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.8,
            price=50000.0,
            atr=1000.0,
            current_time=1000.0
        )
        assert allowed is True
        
        # Second signal different side
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="SHORT",
            confidence=0.7,
            price=50100.0,
            atr=1000.0,
            current_time=1001.0
        )
        assert allowed is True
        assert reason == "allowed"
    
    def test_should_allow_signal_flat_decision(self):
        """Test that FLAT decisions don't interfere with other decisions."""
        gate = SignalGate({'gates': {'nms_window_bars': 3, 'cooldown_seconds': 1}})
        
        # First signal
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.8,
            price=50000.0,
            atr=1000.0,
            current_time=1000.0
        )
        assert allowed is True
        
        # FLAT signal
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="FLAT",
            confidence=0.5,
            price=50100.0,
            atr=1000.0,
            current_time=1001.0
        )
        assert allowed is True
        assert reason == "allowed"
        
        # Another LONG signal should still be allowed
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.9,
            price=50200.0,
            atr=1000.0,
            current_time=1002.0
        )
        assert allowed is True
        assert reason == "allowed"
    
    def test_get_signal_stats(self):
        """Test signal statistics calculation."""
        gate = SignalGate({'gates': {'cooldown_seconds': 1}})
        
        # Add some signals
        gate.should_allow_signal("BTCUSDT", "LONG", 0.8, 50000.0, 1000.0, 1000.0)
        gate.should_allow_signal("BTCUSDT", "SHORT", 0.7, 50100.0, 1000.0, 1001.0)
        gate.should_allow_signal("BTCUSDT", "LONG", 0.9, 50200.0, 1000.0, 1002.0)
        
        stats = gate.get_signal_stats("BTCUSDT")
        
        assert stats['total_signals'] == 3
        assert stats['long_signals'] == 2
        assert stats['short_signals'] == 1
        assert stats['avg_confidence'] == pytest.approx(0.8, rel=1e-3)
        assert stats['last_signal_time'] == 1002.0
    
    def test_get_signal_stats_empty(self):
        """Test signal statistics for symbol with no signals."""
        gate = SignalGate({})
        
        stats = gate.get_signal_stats("BTCUSDT")
        
        assert stats['total_signals'] == 0
        assert stats['long_signals'] == 0
        assert stats['short_signals'] == 0
        assert stats['avg_confidence'] == 0.0
        assert stats['last_signal_time'] == 0.0
    
    def test_clear_history_symbol(self):
        """Test clearing history for specific symbol."""
        gate = SignalGate({})
        
        # Add signals for multiple symbols
        gate.should_allow_signal("BTCUSDT", "LONG", 0.8, 50000.0, 1000.0, 1000.0)
        gate.should_allow_signal("ETHUSDT", "SHORT", 0.7, 3000.0, 100.0, 1001.0)
        
        # Clear BTCUSDT history
        gate.clear_history("BTCUSDT")
        
        assert "BTCUSDT" not in gate.signal_history
        assert "ETHUSDT" in gate.signal_history
        assert len(gate.signal_history["ETHUSDT"]) == 1
    
    def test_clear_history_all(self):
        """Test clearing all history."""
        gate = SignalGate({})
        
        # Add signals for multiple symbols
        gate.should_allow_signal("BTCUSDT", "LONG", 0.8, 50000.0, 1000.0, 1000.0)
        gate.should_allow_signal("ETHUSDT", "SHORT", 0.7, 3000.0, 100.0, 1001.0)
        
        # Clear all history
        gate.clear_history()
        
        assert gate.signal_history == {}
        assert gate.last_signal_time == {}
        assert gate.last_decision == {}
        assert gate.last_price == {}
    
    def test_signal_record_creation(self):
        """Test SignalRecord creation."""
        record = SignalRecord(
            timestamp=1000.0,
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.8,
            price=50000.0,
            atr=1000.0
        )
        
        assert record.timestamp == 1000.0
        assert record.symbol == "BTCUSDT"
        assert record.decision == "LONG"
        assert record.confidence == 0.8
        assert record.price == 50000.0
        assert record.atr == 1000.0


class TestSignalGateFactory:
    """Test signal gate factory functions."""
    
    def test_create_signal_gate(self):
        """Test signal gate factory function."""
        settings = {
            'gates': {
                'nms_window_bars': 5,
                'min_flip_distance_atr': 0.8,
                'cooldown_seconds': 300
            }
        }
        
        gate = create_signal_gate(settings)
        
        assert isinstance(gate, SignalGate)
        assert gate.nms_window_bars == 5
        assert gate.min_flip_distance_atr == 0.8
        assert gate.cooldown_seconds == 300
    
    def test_apply_signal_gate(self):
        """Test apply signal gate function."""
        gate = SignalGate({})
        
        allowed, reason = apply_signal_gate(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.8,
            price=50000.0,
            atr=1000.0,
            gate=gate
        )
        
        assert allowed is True
        assert reason == "allowed"


class TestSignalGateEdgeCases:
    """Test signal gate edge cases."""
    
    def test_zero_atr(self):
        """Test behavior with zero ATR."""
        gate = SignalGate({'gates': {'min_flip_distance_atr': 0.5, 'cooldown_seconds': 1}})
        
        # First signal
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.8,
            price=50000.0,
            atr=0.0,
            current_time=1000.0
        )
        assert allowed is True
        
        # Second signal opposite direction
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="SHORT",
            confidence=0.8,
            price=50100.0,
            atr=0.0,
            current_time=1001.0
        )
        assert allowed is True  # Should allow when ATR is zero
    
    def test_negative_atr(self):
        """Test behavior with negative ATR."""
        gate = SignalGate({'gates': {'min_flip_distance_atr': 0.5, 'cooldown_seconds': 1}})
        
        # First signal
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.8,
            price=50000.0,
            atr=-1000.0,
            current_time=1000.0
        )
        assert allowed is True
        
        # Second signal opposite direction
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="SHORT",
            confidence=0.8,
            price=50100.0,
            atr=-1000.0,
            current_time=1001.0
        )
        assert allowed is True  # Should allow when ATR is negative
    
    def test_extreme_confidence_values(self):
        """Test behavior with extreme confidence values."""
        gate = SignalGate({'gates': {'cooldown_seconds': 1}})
        
        # Test with very low confidence
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.0,
            price=50000.0,
            atr=1000.0,
            current_time=1000.0
        )
        assert allowed is True
        
        # Test with very high confidence
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=1.0,
            price=50100.0,
            atr=1000.0,
            current_time=1001.0
        )
        assert allowed is True
    
    def test_same_price_flip_flop(self):
        """Test flip-flop guard with same price."""
        gate = SignalGate({'gates': {'min_flip_distance_atr': 0.5, 'cooldown_seconds': 1}})
        
        # First signal
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="LONG",
            confidence=0.8,
            price=50000.0,
            atr=1000.0,
            current_time=1000.0
        )
        assert allowed is True
        
        # Second signal opposite direction with same price
        allowed, reason = gate.should_allow_signal(
            symbol="BTCUSDT",
            decision="SHORT",
            confidence=0.8,
            price=50000.0,
            atr=1000.0,
            current_time=1001.0
        )
        assert allowed is False
        assert reason == "flip_flop_guard"


if __name__ == "__main__":
    pytest.main([__file__])
