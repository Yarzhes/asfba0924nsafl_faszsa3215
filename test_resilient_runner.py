#!/usr/bin/env python3
"""
Test script for the resilient signal runner.

This script tests the key improvements:
1. Per-symbol isolation and cooldown
2. TF-specific warmup enforcement
3. Trader-focused Telegram messages
4. Resilient error handling
"""

import asyncio
import time
from unittest.mock import Mock, patch
from loguru import logger

# Mock the imports to avoid actual WebSocket connections
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


def test_symbol_state():
    """Test per-symbol state tracking."""
    state = SymbolState()
    
    # Test initial state
    assert state.last_signal_ts == 0.0
    assert state.last_signal_side == "FLAT"
    assert state.consecutive_signals == 0
    
    # Test state updates
    current_time = time.time()
    state.last_signal_ts = current_time
    state.last_signal_side = "LONG"
    state.consecutive_signals = 1
    
    assert state.last_signal_ts == current_time
    assert state.last_signal_side == "LONG"
    assert state.consecutive_signals == 1
    
    print("✅ SymbolState tests passed")


def test_sl_tp_calculation():
    """Test SL/TP calculation logic."""
    from ultra_signals.transport.telegram import _calculate_sl_tp
    
    # Test LONG position
    entry_price = 50000.0
    atr = 1000.0
    settings = {'execution': {'sl_atr_multiplier': 1.5}}
    
    result = _calculate_sl_tp(entry_price, "LONG", atr, settings)
    
    expected_sl = entry_price - (1.5 * atr)  # 48500
    expected_tp1 = entry_price + (1.0 * (entry_price - expected_sl))  # 51500
    expected_tp2 = entry_price + (1.5 * (entry_price - expected_sl))  # 52250
    expected_tp3 = entry_price + (2.0 * (entry_price - expected_sl))  # 53000
    
    assert abs(result['stop_loss'] - expected_sl) < 0.01
    assert abs(result['tp1'] - expected_tp1) < 0.01
    assert abs(result['tp2'] - expected_tp2) < 0.01
    assert abs(result['tp3'] - expected_tp3) < 0.01
    
    # Test SHORT position
    result = _calculate_sl_tp(entry_price, "SHORT", atr, settings)
    
    expected_sl = entry_price + (1.5 * atr)  # 51500
    expected_tp1 = entry_price - (1.0 * (expected_sl - entry_price))  # 48500
    expected_tp2 = entry_price - (1.5 * (expected_sl - entry_price))  # 47750
    expected_tp3 = entry_price - (2.0 * (expected_sl - entry_price))  # 47000
    
    assert abs(result['stop_loss'] - expected_sl) < 0.01
    assert abs(result['tp1'] - expected_tp1) < 0.01
    assert abs(result['tp2'] - expected_tp2) < 0.01
    assert abs(result['tp3'] - expected_tp3) < 0.01
    
    print("✅ SL/TP calculation tests passed")


def test_cooldown_logic():
    """Test cooldown and isolation logic."""
    # Mock settings
    settings = Mock()
    settings.runtime.get.return_value = 60.0  # min_signal_interval_sec
    settings.runtime.min_confidence = 0.65
    
    # Create runner instance
    runner = ResilientSignalRunner(settings)
    
    current_time = time.time()
    
    # Test minimum interval
    symbol = "BTCUSDT"
    state = runner.symbol_states[symbol]
    state.last_signal_ts = current_time - 30.0  # 30 seconds ago
    
    # Mock decision
    decision = Mock()
    decision.confidence = 0.8
    decision.decision = "LONG"
    
    # Should not send signal (too soon)
    should_send = runner._should_send_signal(symbol, decision, current_time)
    assert not should_send
    
    # Test after minimum interval
    state.last_signal_ts = current_time - 120.0  # 2 minutes ago
    should_send = runner._should_send_signal(symbol, decision, current_time)
    assert should_send
    
    # Test consecutive signals cooldown
    state.consecutive_signals = 3
    state.cooldown_until = current_time + 60.0  # Cooldown active
    
    should_send = runner._should_send_signal(symbol, decision, current_time)
    assert not should_send
    
    # Test after cooldown expires
    state.cooldown_until = current_time - 60.0  # Cooldown expired
    should_send = runner._should_send_signal(symbol, decision, current_time)
    assert should_send
    
    # Test confidence threshold
    decision.confidence = 0.5  # Below threshold
    should_send = runner._should_send_signal(symbol, decision, current_time)
    assert not should_send
    
    print("✅ Cooldown logic tests passed")


def test_timeframe_ready_check():
    """Test timeframe readiness checking."""
    settings = Mock()
    settings.features.warmup_periods = 200
    
    runner = ResilientSignalRunner(settings)
    
    # Mock feature store
    mock_ohlcv = Mock()
    mock_ohlcv.__len__ = lambda self: 150  # Insufficient data
    
    runner.feature_store.get_ohlcv = Mock(return_value=mock_ohlcv)
    
    # Test insufficient data
    is_ready = runner._is_timeframe_ready("BTCUSDT", "15m")
    assert not is_ready
    
    # Test sufficient data
    mock_ohlcv.__len__ = lambda self: 250  # Sufficient data
    is_ready = runner._is_timeframe_ready("BTCUSDT", "15m")
    assert is_ready
    
    # Test that timeframe is marked as ready
    assert "15m" in runner.ready_timeframes["BTCUSDT"]
    
    print("✅ Timeframe readiness tests passed")


async def test_message_formatting():
    """Test trader-focused message formatting."""
    from ultra_signals.transport.telegram import format_message
    
    # Mock decision
    decision = Mock()
    decision.decision = "LONG"
    decision.symbol = "BTCUSDT"
    decision.tf = "5m"
    decision.confidence = 0.75
    
    # Mock vote detail with ATR
    decision.vote_detail = {
        'risk_model': {
            'entry_price': 50000.0,
            'atr': 1000.0
        }
    }
    
    # Mock settings
    settings = {
        'execution': {
            'sl_atr_multiplier': 1.5,
            'default_leverage': 10
        },
        'position_sizing': {
            'max_risk_pct': 0.01
        }
    }
    
    # Format message
    message = format_message(decision, settings)
    
    print(f"Generated message:\n{message}")
    
    # Check that message contains required fields (more flexible assertions)
    assert "LONG" in message
    assert "BTCUSDT" in message
    assert "Confidence:" in message
    assert "Entry:" in message
    assert "Stop Loss:" in message
    assert "TP1:" in message
    assert "TP2:" in message
    assert "TP3:" in message
    assert "Leverage:" in message
    assert "Risk:" in message
    assert "R:R" in message
    assert "Time:" in message
    
    print("✅ Message formatting tests passed")


def main():
    """Run all tests."""
    print("🧪 Testing Ultra Signals Resilient Runner...")
    
    try:
        test_symbol_state()
        test_sl_tp_calculation()
        test_cooldown_logic()
        test_timeframe_ready_check()
        
        # Run async test
        asyncio.run(test_message_formatting())
        
        print("\n🎉 All tests passed! The resilient runner is ready for deployment.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
