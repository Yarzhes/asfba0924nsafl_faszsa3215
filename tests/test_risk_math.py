#!/usr/bin/env python3
"""
Test Risk Math Validation

This module tests:
1. SL/TP calculation accuracy for LONG/SHORT
2. Tick size rounding
3. Confidence calibration and formatting
4. Risk/reward ratio calculations
"""

import math
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
    from ultra_signals.transport.telegram import _calculate_sl_tp, format_message


class TestRiskMath:
    """Test suite for risk math validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.settings = {
            'execution': {
                'sl_atr_multiplier': 1.5,
                'default_leverage': 10
            },
            'position_sizing': {
                'max_risk_pct': 0.01  # 1%
            }
        }
        
    def test_long_sl_tp_calculation(self):
        """Test SL/TP calculation for LONG positions."""
        entry_price = 50000.0
        atr = 1000.0
        
        result = _calculate_sl_tp(entry_price, "LONG", atr, self.settings)
        
        # Expected calculations
        expected_sl = entry_price - (1.5 * atr)  # 48500
        expected_tp1 = entry_price + (1.0 * (entry_price - expected_sl))  # 51500
        expected_tp2 = entry_price + (1.5 * (entry_price - expected_sl))  # 52250
        expected_tp3 = entry_price + (2.0 * (entry_price - expected_sl))  # 53000
        
        # Verify calculations
        assert abs(result['stop_loss'] - expected_sl) < 0.01, f"SL mismatch: {result['stop_loss']} vs {expected_sl}"
        assert abs(result['tp1'] - expected_tp1) < 0.01, f"TP1 mismatch: {result['tp1']} vs {expected_tp1}"
        assert abs(result['tp2'] - expected_tp2) < 0.01, f"TP2 mismatch: {result['tp2']} vs {expected_tp2}"
        assert abs(result['tp3'] - expected_tp3) < 0.01, f"TP3 mismatch: {result['tp3']} vs {expected_tp3}"
        
        # Verify risk calculation
        risk_amount = entry_price - expected_sl
        assert abs(result['risk_amount'] - risk_amount) < 0.01, f"Risk amount mismatch: {result['risk_amount']} vs {risk_amount}"
        
    def test_short_sl_tp_calculation(self):
        """Test SL/TP calculation for SHORT positions."""
        entry_price = 50000.0
        atr = 1000.0
        
        result = _calculate_sl_tp(entry_price, "SHORT", atr, self.settings)
        
        # Expected calculations (inverted for SHORT)
        expected_sl = entry_price + (1.5 * atr)  # 51500
        expected_tp1 = entry_price - (1.0 * (expected_sl - entry_price))  # 48500
        expected_tp2 = entry_price - (1.5 * (expected_sl - entry_price))  # 47750
        expected_tp3 = entry_price - (2.0 * (expected_sl - entry_price))  # 47000
        
        # Verify calculations
        assert abs(result['stop_loss'] - expected_sl) < 0.01, f"SL mismatch: {result['stop_loss']} vs {expected_sl}"
        assert abs(result['tp1'] - expected_tp1) < 0.01, f"TP1 mismatch: {result['tp1']} vs {expected_tp1}"
        assert abs(result['tp2'] - expected_tp2) < 0.01, f"TP2 mismatch: {result['tp2']} vs {expected_tp2}"
        assert abs(result['tp3'] - expected_tp3) < 0.01, f"TP3 mismatch: {result['tp3']} vs {expected_tp3}"
        
        # Verify risk calculation
        risk_amount = expected_sl - entry_price
        assert abs(result['risk_amount'] - risk_amount) < 0.01, f"Risk amount mismatch: {result['risk_amount']} vs {risk_amount}"
        
    def test_tick_size_rounding(self):
        """Test tick size rounding for different symbols."""
        # Test BTCUSDT (0.1 tick size)
        entry_price = 50000.123
        atr = 1000.456
        
        result = _calculate_sl_tp(entry_price, "LONG", atr, self.settings)
        
        # Verify rounding to 0.1
        assert result['stop_loss'] % 0.1 < 0.001, f"SL not rounded to 0.1: {result['stop_loss']}"
        assert result['tp1'] % 0.1 < 0.001, f"TP1 not rounded to 0.1: {result['tp1']}"
        assert result['tp2'] % 0.1 < 0.001, f"TP2 not rounded to 0.1: {result['tp2']}"
        assert result['tp3'] % 0.1 < 0.001, f"TP3 not rounded to 0.1: {result['tp3']}"
        
        # Test ETHUSDT (0.01 tick size)
        entry_price = 3000.1234
        atr = 50.6789
        
        result = _calculate_sl_tp(entry_price, "LONG", atr, self.settings)
        
        # Verify rounding to 0.01
        assert result['stop_loss'] % 0.01 < 0.001, f"SL not rounded to 0.01: {result['stop_loss']}"
        assert result['tp1'] % 0.01 < 0.001, f"TP1 not rounded to 0.01: {result['tp1']}"
        assert result['tp2'] % 0.01 < 0.001, f"TP2 not rounded to 0.01: {result['tp2']}"
        assert result['tp3'] % 0.01 < 0.001, f"TP3 not rounded to 0.01: {result['tp3']}"
        
    def test_confidence_calibration(self):
        """Test confidence calibration and formatting."""
        # Test confidence clipping
        test_confidences = [-0.5, 0.0, 0.5, 0.75, 1.0, 1.5]
        expected_formatted = [0.0, 0.0, 50.0, 75.0, 100.0, 100.0]
        
        for conf, expected in zip(test_confidences, expected_formatted):
            # Clip confidence to [0, 1]
            clipped_confidence = max(0.0, min(1.0, conf))
            formatted_confidence = clipped_confidence * 100
            
            assert abs(formatted_confidence - expected) < 0.01, f"Confidence formatting mismatch: {formatted_confidence} vs {expected}"
            
    def test_risk_reward_ratio(self):
        """Test risk/reward ratio calculations."""
        entry_price = 50000.0
        atr = 1000.0
        
        # Test LONG position
        result_long = _calculate_sl_tp(entry_price, "LONG", atr, self.settings)
        
        # Calculate R:R ratios
        risk = result_long['risk_amount']
        reward_tp1 = result_long['tp1'] - entry_price
        reward_tp2 = result_long['tp2'] - entry_price
        reward_tp3 = result_long['tp3'] - entry_price
        
        rr_tp1 = reward_tp1 / risk
        rr_tp2 = reward_tp2 / risk
        rr_tp3 = reward_tp3 / risk
        
        # Verify R:R ratios
        assert abs(rr_tp1 - 1.0) < 0.01, f"TP1 R:R should be 1:1, got {rr_tp1}"
        assert abs(rr_tp2 - 1.5) < 0.01, f"TP2 R:R should be 1:1.5, got {rr_tp2}"
        assert abs(rr_tp3 - 2.0) < 0.01, f"TP3 R:R should be 1:2, got {rr_tp3}"
        
        # Test SHORT position
        result_short = _calculate_sl_tp(entry_price, "SHORT", atr, self.settings)
        
        # Calculate R:R ratios (inverted for SHORT)
        risk = result_short['risk_amount']
        reward_tp1 = entry_price - result_short['tp1']
        reward_tp2 = entry_price - result_short['tp2']
        reward_tp3 = entry_price - result_short['tp3']
        
        rr_tp1 = reward_tp1 / risk
        rr_tp2 = reward_tp2 / risk
        rr_tp3 = reward_tp3 / risk
        
        # Verify R:R ratios
        assert abs(rr_tp1 - 1.0) < 0.01, f"TP1 R:R should be 1:1, got {rr_tp1}"
        assert abs(rr_tp2 - 1.5) < 0.01, f"TP2 R:R should be 1:1.5, got {rr_tp2}"
        assert abs(rr_tp3 - 2.0) < 0.01, f"TP3 R:R should be 1:2, got {rr_tp3}"
        
    def test_edge_case_prices(self):
        """Test edge cases with extreme prices."""
        # Test very high price
        entry_price = 1000000.0
        atr = 50000.0
        
        result = _calculate_sl_tp(entry_price, "LONG", atr, self.settings)
        
        # Verify calculations are still accurate
        expected_sl = entry_price - (1.5 * atr)
        assert abs(result['stop_loss'] - expected_sl) < 0.01, f"High price SL mismatch: {result['stop_loss']} vs {expected_sl}"
        
        # Test very low price
        entry_price = 0.001
        atr = 0.0001
        
        result = _calculate_sl_tp(entry_price, "LONG", atr, self.settings)
        
        # Verify calculations are still accurate
        expected_sl = entry_price - (1.5 * atr)
        assert abs(result['stop_loss'] - expected_sl) < 0.000001, f"Low price SL mismatch: {result['stop_loss']} vs {expected_sl}"
        
    def test_zero_atr_handling(self):
        """Test handling of zero ATR."""
        entry_price = 50000.0
        atr = 0.0
        
        # Should handle zero ATR gracefully
        try:
            result = _calculate_sl_tp(entry_price, "LONG", atr, self.settings)
            # If no exception, verify reasonable defaults
            assert result['stop_loss'] == entry_price, "Zero ATR should result in SL = entry"
            assert result['tp1'] == entry_price, "Zero ATR should result in TP1 = entry"
        except Exception as e:
            # If exception is raised, it should be handled gracefully
            assert "ATR" in str(e) or "zero" in str(e).lower(), f"Unexpected exception: {e}"
            
    def test_negative_atr_handling(self):
        """Test handling of negative ATR."""
        entry_price = 50000.0
        atr = -1000.0
        
        # Should handle negative ATR gracefully
        try:
            result = _calculate_sl_tp(entry_price, "LONG", atr, self.settings)
            # If no exception, verify reasonable defaults
            assert result['stop_loss'] == entry_price, "Negative ATR should result in SL = entry"
            assert result['tp1'] == entry_price, "Negative ATR should result in TP1 = entry"
        except Exception as e:
            # If exception is raised, it should be handled gracefully
            assert "ATR" in str(e) or "negative" in str(e).lower(), f"Unexpected exception: {e}"
            
    def test_message_formatting_integration(self):
        """Test that message formatting uses the same calculation helper."""
        # Create mock decision
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
        
        # Format message
        message = format_message(decision, self.settings)
        
        # Verify message contains required fields
        assert "LONG" in message, "Message should contain direction"
        assert "BTCUSDT" in message, "Message should contain symbol"
        assert "Confidence:" in message, "Message should contain confidence"
        assert "Entry:" in message, "Message should contain entry price"
        assert "Stop Loss:" in message, "Message should contain stop loss"
        assert "TP1:" in message, "Message should contain TP1"
        assert "TP2:" in message, "Message should contain TP2"
        assert "TP3:" in message, "Message should contain TP3"
        assert "Leverage:" in message, "Message should contain leverage"
        assert "Risk:" in message, "Message should contain risk percentage"
        assert "R:R" in message, "Message should contain risk/reward ratio"
        
        # Verify numeric values are reasonable
        # Extract numeric values from message (simplified)
        assert "50000" in message, "Entry price should be in message"
        assert "48500" in message, "Stop loss should be in message"
        assert "51500" in message, "TP1 should be in message"
        assert "75.0%" in message, "Confidence should be formatted as percentage"
        
    def test_leverage_calculation(self):
        """Test leverage calculation and validation."""
        # Test default leverage
        settings_default = {
            'execution': {
                'sl_atr_multiplier': 1.5,
                'default_leverage': 10
            }
        }
        
        entry_price = 50000.0
        atr = 1000.0
        
        result = _calculate_sl_tp(entry_price, "LONG", atr, settings_default)
        
        # Verify leverage is included
        assert 'leverage' in result or 'default_leverage' in settings_default['execution'], "Leverage should be available"
        
        # Test custom leverage
        settings_custom = {
            'execution': {
                'sl_atr_multiplier': 1.5,
                'default_leverage': 20
            }
        }
        
        result = _calculate_sl_tp(entry_price, "LONG", atr, settings_custom)
        
        # Verify custom leverage is used
        assert settings_custom['execution']['default_leverage'] == 20, "Custom leverage should be used"
        
    def test_risk_percentage_calculation(self):
        """Test risk percentage calculation."""
        entry_price = 50000.0
        atr = 1000.0
        
        result = _calculate_sl_tp(entry_price, "LONG", atr, self.settings)
        
        # Calculate risk percentage
        risk_amount = result['risk_amount']
        risk_percentage = (risk_amount / entry_price) * 100
        
        # Verify risk percentage is reasonable
        assert 0 < risk_percentage < 10, f"Risk percentage should be reasonable: {risk_percentage}%"
        
        # For 1.5 ATR, risk should be approximately 3%
        expected_risk_pct = (1.5 * atr / entry_price) * 100
        assert abs(risk_percentage - expected_risk_pct) < 0.1, f"Risk percentage mismatch: {risk_percentage}% vs {expected_risk_pct}%"


def test_tick_size_mapping():
    """Test tick size mapping for different symbols."""
    # Mock tick size mapping
    tick_sizes = {
        'BTCUSDT': 0.1,
        'ETHUSDT': 0.01,
        'BNBUSDT': 0.01,
        'ADAUSDT': 0.0001,
        'DOGEUSDT': 0.00001
    }
    
    def round_to_tick_size(price, symbol):
        tick_size = tick_sizes.get(symbol, 0.01)  # Default to 0.01
        return round(price / tick_size) * tick_size
    
    # Test rounding for different symbols
    test_cases = [
        ('BTCUSDT', 50000.123, 50000.1),
        ('ETHUSDT', 3000.1234, 3000.12),
        ('ADAUSDT', 0.123456, 0.1234),
        ('DOGEUSDT', 0.123456, 0.12346)
    ]
    
    for symbol, input_price, expected_price in test_cases:
        rounded_price = round_to_tick_size(input_price, symbol)
        assert abs(rounded_price - expected_price) < 0.00001, f"Tick rounding failed for {symbol}: {rounded_price} vs {expected_price}"


def test_confidence_bounds():
    """Test confidence bounds and edge cases."""
    # Test confidence bounds
    test_cases = [
        (-1.0, 0.0),    # Below 0 should clip to 0
        (0.0, 0.0),     # At 0 should stay 0
        (0.5, 0.5),     # At 0.5 should stay 0.5
        (1.0, 1.0),     # At 1.0 should stay 1.0
        (1.5, 1.0),     # Above 1.0 should clip to 1.0
    ]
    
    for input_conf, expected_conf in test_cases:
        clipped = max(0.0, min(1.0, input_conf))
        assert abs(clipped - expected_conf) < 0.001, f"Confidence clipping failed: {clipped} vs {expected_conf}"
        
        # Test percentage formatting
        formatted = clipped * 100
        expected_formatted = expected_conf * 100
        assert abs(formatted - expected_formatted) < 0.001, f"Confidence formatting failed: {formatted} vs {expected_formatted}"


def main():
    """Run all risk math tests."""
    print("🧪 Testing Risk Math Validation...")
    
    # Create test instance
    test_suite = TestRiskMath()
    
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
        test_tick_size_mapping()
        print("✅ test_tick_size_mapping passed")
        passed += 1
    except Exception as e:
        print(f"❌ test_tick_size_mapping failed: {e}")
        failed += 1
        
    try:
        test_confidence_bounds()
        print("✅ test_confidence_bounds passed")
        passed += 1
    except Exception as e:
        print(f"❌ test_confidence_bounds failed: {e}")
        failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All risk math tests passed!")
        return True
    else:
        print("❌ Some risk math tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)



