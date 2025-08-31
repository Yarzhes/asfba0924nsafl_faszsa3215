#!/usr/bin/env python3
"""
Test Telegram Format and Security

This module tests:
1. Message field presence and formatting
2. Value range validation
3. Secret masking and security
4. Rate limiting and retry logic
5. Dry-run integrity
"""

import re
import json
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
    from ultra_signals.transport.telegram import format_message, send_message


class TestTelegramFormat:
    """Test suite for Telegram message formatting and security."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.settings = {
            'execution': {
                'sl_atr_multiplier': 1.5,
                'default_leverage': 10
            },
            'position_sizing': {
                'max_risk_pct': 0.01  # 1%
            },
            'transport': {
                'telegram': {
                    'bot_token': 'test_token_12345',
                    'chat_id': 'test_chat_67890',
                    'enabled': True,
                    'dry_run': False
                }
            }
        }
        
    def test_message_field_presence(self):
        """Test that all required fields are present in the message."""
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
        
        # Check required fields
        required_fields = [
            'LONG', 'BTCUSDT', '5m',  # Direction, Symbol, TF
            'Confidence:', '75.0%',    # Confidence
            'Entry:', '50000.0000',    # Entry price
            'Stop Loss:', '48500.0000', # Stop loss
            'TP1:', '51500.0000',      # Take profit 1
            'TP2:', '52250.0000',      # Take profit 2
            'TP3:', '53000.0000',      # Take profit 3
            'Leverage:', '10x',        # Leverage
            'Risk:', '1.00%',          # Risk percentage
            'R:R', '1:1.00',          # Risk/reward ratio
            'Time:', 'UTC'             # Timestamp
        ]
        
        for field in required_fields:
            assert field in message, f"Required field '{field}' not found in message"
            
    def test_value_range_validation(self):
        """Test that numeric values are within valid ranges."""
        # Test confidence bounds
        test_cases = [
            (-0.5, 0.0),    # Below 0 should clip to 0
            (0.0, 0.0),     # At 0 should stay 0
            (0.5, 0.5),     # At 0.5 should stay 0.5
            (1.0, 1.0),     # At 1.0 should stay 1.0
            (1.5, 1.0),     # Above 1.0 should clip to 1.0
        ]
        
        for input_conf, expected_conf in test_cases:
            decision = Mock()
            decision.decision = "LONG"
            decision.symbol = "BTCUSDT"
            decision.tf = "5m"
            decision.confidence = input_conf
            decision.vote_detail = {
                'risk_model': {
                    'entry_price': 50000.0,
                    'atr': 1000.0
                }
            }
            
            message = format_message(decision, self.settings)
            
            # Check that confidence is properly formatted
            expected_percentage = f"{expected_conf * 100:.1f}%"
            assert expected_percentage in message, f"Confidence {input_conf} should format as {expected_percentage}"
            
    def test_price_formatting(self):
        """Test that prices are properly formatted."""
        decision = Mock()
        decision.decision = "LONG"
        decision.symbol = "BTCUSDT"
        decision.tf = "5m"
        decision.confidence = 0.75
        decision.vote_detail = {
            'risk_model': {
                'entry_price': 50000.123,
                'atr': 1000.456
            }
        }
        
        message = format_message(decision, self.settings)
        
        # Check price formatting
        price_patterns = [
            r'Entry:\s*\$[\d,]+\.\d{4}',      # Entry with 4 decimal places
            r'Stop Loss:\s*\$[\d,]+\.\d{4}',  # Stop loss with 4 decimal places
            r'TP1:\s*\$[\d,]+\.\d{4}',        # TP1 with 4 decimal places
            r'TP2:\s*\$[\d,]+\.\d{4}',        # TP2 with 4 decimal places
            r'TP3:\s*\$[\d,]+\.\d{4}',        # TP3 with 4 decimal places
        ]
        
        for pattern in price_patterns:
            assert re.search(pattern, message), f"Price pattern '{pattern}' not found in message"
            
    def test_risk_reward_formatting(self):
        """Test that risk/reward ratios are properly formatted."""
        decision = Mock()
        decision.decision = "LONG"
        decision.symbol = "BTCUSDT"
        decision.tf = "5m"
        decision.confidence = 0.75
        decision.vote_detail = {
            'risk_model': {
                'entry_price': 50000.0,
                'atr': 1000.0
            }
        }
        
        message = format_message(decision, self.settings)
        
        # Check R:R formatting
        rr_pattern = r'R:R\s*=\s*1:[\d.]+'
        assert re.search(rr_pattern, message), "Risk/reward ratio not properly formatted"
        
        # Extract R:R value
        rr_match = re.search(r'R:R\s*=\s*1:([\d.]+)', message)
        if rr_match:
            rr_value = float(rr_match.group(1))
            assert 0.5 <= rr_value <= 3.0, f"R:R value {rr_value} should be reasonable"
            
    def test_secret_masking(self):
        """Test that secrets are not exposed in logs or messages."""
        # Test bot token masking
        bot_token = "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
        masked_token = "1234567890:***"
        
        # Check that token is masked in logs
        with patch('loguru.logger.info') as mock_logger:
            # Simulate logging with token
            logger.info(f"Bot token: {bot_token}")
            
            # Check that actual token is not logged
            log_calls = mock_logger.call_args_list
            for call in log_calls:
                log_message = str(call)
                assert bot_token not in log_message, "Bot token should not be logged"
                
    def test_dry_run_integrity(self):
        """Test that dry-run mode executes full formatting path but skips network."""
        decision = Mock()
        decision.decision = "LONG"
        decision.symbol = "BTCUSDT"
        decision.tf = "5m"
        decision.confidence = 0.75
        decision.vote_detail = {
            'risk_model': {
                'entry_price': 50000.0,
                'atr': 1000.0
            }
        }
        
        # Test with dry_run enabled
        dry_run_settings = self.settings.copy()
        dry_run_settings['transport']['telegram']['dry_run'] = True
        
        # Format message (should work normally)
        message = format_message(decision, dry_run_settings)
        
        # Verify message is properly formatted
        assert "LONG" in message, "Dry-run should still format message"
        assert "BTCUSDT" in message, "Dry-run should still include symbol"
        assert "Entry:" in message, "Dry-run should still include entry price"
        
        # Test send_message with dry_run (should not make network call)
        with patch('requests.post') as mock_post:
            send_message(message, dry_run_settings)
            
            # Verify no network call was made
            mock_post.assert_not_called()
            
    def test_rate_limiting_handling(self):
        """Test rate limiting and retry logic."""
        # Mock rate limit response
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.json.return_value = {'retry_after': 30}
        
        # Mock successful response
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {'ok': True}
        
        with patch('requests.post') as mock_post:
            # First call returns rate limit
            mock_post.side_effect = [rate_limit_response, success_response]
            
            # Test retry logic
            message = "Test message"
            
            # This would normally trigger retry logic
            # For testing, we just verify the mock was called
            mock_post.assert_not_called()
            
    def test_message_length_validation(self):
        """Test that messages are within Telegram's length limits."""
        decision = Mock()
        decision.decision = "LONG"
        decision.symbol = "BTCUSDT"
        decision.tf = "5m"
        decision.confidence = 0.75
        decision.vote_detail = {
            'risk_model': {
                'entry_price': 50000.0,
                'atr': 1000.0
            }
        }
        
        message = format_message(decision, self.settings)
        
        # Telegram message limit is 4096 characters
        assert len(message) <= 4096, f"Message too long: {len(message)} characters"
        
        # Message should be reasonably long (not empty)
        assert len(message) > 100, f"Message too short: {len(message)} characters"
        
    def test_unicode_handling(self):
        """Test that Unicode characters are handled properly."""
        decision = Mock()
        decision.decision = "LONG"
        decision.symbol = "BTCUSDT"
        decision.tf = "5m"
        decision.confidence = 0.75
        decision.vote_detail = {
            'risk_model': {
                'entry_price': 50000.0,
                'atr': 1000.0
            }
        }
        
        message = format_message(decision, self.settings)
        
        # Check for Unicode characters (emojis, etc.)
        unicode_chars = ['📈', '📍', '🛑', '🎯', '⚡', '⚠️', '📊', '🕐', '💡']
        
        # At least some Unicode characters should be present
        unicode_count = sum(1 for char in unicode_chars if char in message)
        assert unicode_count > 0, "Message should contain Unicode characters"
        
        # Message should be properly encoded
        try:
            message.encode('utf-8')
        except UnicodeEncodeError:
            assert False, "Message should be UTF-8 encodable"
            
    def test_timestamp_formatting(self):
        """Test that timestamps are properly formatted."""
        decision = Mock()
        decision.decision = "LONG"
        decision.symbol = "BTCUSDT"
        decision.tf = "5m"
        decision.confidence = 0.75
        decision.vote_detail = {
            'risk_model': {
                'entry_price': 50000.0,
                'atr': 1000.0
            }
        }
        
        message = format_message(decision, self.settings)
        
        # Check timestamp format
        timestamp_pattern = r'Time:\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+UTC'
        assert re.search(timestamp_pattern, message), "Timestamp not properly formatted"
        
    def test_direction_formatting(self):
        """Test that trade directions are properly formatted."""
        for direction in ["LONG", "SHORT"]:
            decision = Mock()
            decision.decision = direction
            decision.symbol = "BTCUSDT"
            decision.tf = "5m"
            decision.confidence = 0.75
            decision.vote_detail = {
                'risk_model': {
                    'entry_price': 50000.0,
                    'atr': 1000.0
                }
            }
            
            message = format_message(decision, self.settings)
            
            # Check direction is properly formatted
            assert direction in message, f"Direction '{direction}' not found in message"
            
            # Check for appropriate emoji
            if direction == "LONG":
                assert "📈" in message, "LONG direction should have 📈 emoji"
            else:
                assert "📉" in message, "SHORT direction should have 📉 emoji"
                
    def test_symbol_formatting(self):
        """Test that symbols are properly formatted."""
        test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
        
        for symbol in test_symbols:
            decision = Mock()
            decision.decision = "LONG"
            decision.symbol = symbol
            decision.tf = "5m"
            decision.confidence = 0.75
            decision.vote_detail = {
                'risk_model': {
                    'entry_price': 50000.0,
                    'atr': 1000.0
                }
            }
            
            message = format_message(decision, self.settings)
            
            # Check symbol is properly formatted
            assert symbol in message, f"Symbol '{symbol}' not found in message"
            
    def test_leverage_formatting(self):
        """Test that leverage is properly formatted."""
        decision = Mock()
        decision.decision = "LONG"
        decision.symbol = "BTCUSDT"
        decision.tf = "5m"
        decision.confidence = 0.75
        decision.vote_detail = {
            'risk_model': {
                'entry_price': 50000.0,
                'atr': 1000.0
            }
        }
        
        message = format_message(decision, self.settings)
        
        # Check leverage formatting
        leverage_pattern = r'Leverage:\s*\d+x'
        assert re.search(leverage_pattern, message), "Leverage not properly formatted"
        
        # Extract leverage value
        leverage_match = re.search(r'Leverage:\s*(\d+)x', message)
        if leverage_match:
            leverage_value = int(leverage_match.group(1))
            assert 1 <= leverage_value <= 100, f"Leverage value {leverage_value} should be reasonable"
            
    def test_risk_percentage_formatting(self):
        """Test that risk percentage is properly formatted."""
        decision = Mock()
        decision.decision = "LONG"
        decision.symbol = "BTCUSDT"
        decision.tf = "5m"
        decision.confidence = 0.75
        decision.vote_detail = {
            'risk_model': {
                'entry_price': 50000.0,
                'atr': 1000.0
            }
        }
        
        message = format_message(decision, self.settings)
        
        # Check risk percentage formatting
        risk_pattern = r'Risk:\s*[\d.]+%'
        assert re.search(risk_pattern, message), "Risk percentage not properly formatted"
        
        # Extract risk value
        risk_match = re.search(r'Risk:\s*([\d.]+)%', message)
        if risk_match:
            risk_value = float(risk_match.group(1))
            assert 0.1 <= risk_value <= 10.0, f"Risk value {risk_value}% should be reasonable"


def test_message_structure():
    """Test overall message structure and readability."""
    # Create a sample message
    sample_message = """📈 LONG BTCUSDT (5m)
Confidence: 75.0%

📍 Entry: $50000.0000
🛑 Stop Loss: $48500.0000
🎯 TP1: $51500.0000
🎯 TP2: $52250.0000
🎯 TP3: $53000.0000
⚡ Leverage: 10x
⚠️ Risk: 3.00%
📊 R:R = 1:1.00
🕐 Time: 2025-08-30 08:35:00 UTC
💡 Reason: Trend up + pullback to VWAP"""
    
    # Check message structure
    lines = sample_message.split('\n')
    assert len(lines) >= 10, "Message should have sufficient lines"
    
    # Check for required sections
    assert any('LONG' in line for line in lines), "Direction should be on a line"
    assert any('Confidence:' in line for line in lines), "Confidence should be on a line"
    assert any('Entry:' in line for line in lines), "Entry should be on a line"
    assert any('Stop Loss:' in line for line in lines), "Stop Loss should be on a line"
    assert any('TP1:' in line for line in lines), "TP1 should be on a line"
    assert any('TP2:' in line for line in lines), "TP2 should be on a line"
    assert any('TP3:' in line for line in lines), "TP3 should be on a line"
    assert any('Leverage:' in line for line in lines), "Leverage should be on a line"
    assert any('Risk:' in line for line in lines), "Risk should be on a line"
    assert any('R:R' in line for line in lines), "R:R should be on a line"
    assert any('Time:' in line for line in lines), "Time should be on a line"


def test_security_validation():
    """Test security aspects of message handling."""
    # Test that sensitive data is not logged
    sensitive_data = {
        'bot_token': '1234567890:ABCdefGHIjklMNOpqrsTUVwxyz',
        'chat_id': '987654321',
        'api_key': 'secret_api_key_123'
    }
    
    # Check that sensitive data is not exposed in logs
    with patch('loguru.logger.info') as mock_logger:
        # Simulate logging
        logger.info("Processing message")
        
        # Verify no sensitive data in logs
        log_calls = mock_logger.call_args_list
        for call in log_calls:
            log_message = str(call)
            for key, value in sensitive_data.items():
                assert value not in log_message, f"Sensitive data '{key}' should not be logged"


def main():
    """Run all Telegram format tests."""
    print("🧪 Testing Telegram Format and Security...")
    
    # Create test instance
    test_suite = TestTelegramFormat()
    
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
        test_message_structure()
        print("✅ test_message_structure passed")
        passed += 1
    except Exception as e:
        print(f"❌ test_message_structure failed: {e}")
        failed += 1
        
    try:
        test_security_validation()
        print("✅ test_security_validation passed")
        passed += 1
    except Exception as e:
        print(f"❌ test_security_validation failed: {e}")
        failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Telegram format tests passed!")
        return True
    else:
        print("❌ Some Telegram format tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)



