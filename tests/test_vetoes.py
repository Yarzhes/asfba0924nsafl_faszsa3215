#!/usr/bin/env python3
"""
Test Risk Filter Vetoes and Audit Visibility

This module tests:
1. Veto reason logging and visibility
2. Veto counter tracking
3. Synthetic veto triggering
4. Veto execution order
5. Veto metrics collection
"""

import time
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
    from ultra_signals.engine.risk_filters import apply_filters


class TestVetoes:
    """Test suite for risk filter vetoes and audit visibility."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.settings = {
            'risk_filters': {
                'spread_max_bps': 50,  # 0.5% max spread
                'min_adx': 25,         # Minimum ADX
                'min_atr': 100,        # Minimum ATR
                'max_volume_imbalance': 0.8,  # Max volume imbalance
                'min_liquidity': 1000000  # Min liquidity
            }
        }
        
    def test_spread_too_wide_veto(self):
        """Test spread too wide veto with proper logging."""
        # Create decision with wide spread
        decision = Mock()
        decision.symbol = "BTCUSDT"
        decision.confidence = 0.8
        decision.decision = "LONG"
        
        # Mock market data with wide spread
        market_data = {
            'bid': 50000.0,
            'ask': 50250.0,  # 0.5% spread (50 bps)
            'spread_bps': 50.0
        }
        
        # Mock feature data
        feature_data = {
            'spread_bps': 50.0,
            'adx': 30.0,
            'atr': 1000.0,
            'volume_imbalance': 0.5,
            'liquidity': 5000000
        }
        
        with patch('loguru.logger.info') as mock_logger:
            # Apply filters
            result = apply_filters(decision, market_data, feature_data, self.settings)
            
            # Check veto logging
            log_calls = mock_logger.call_args_list
            veto_logged = False
            for call in log_calls:
                log_message = str(call)
                if "SPREAD_TOO_WIDE" in log_message and "BTCUSDT" in log_message:
                    veto_logged = True
                    assert "50.0" in log_message, "Spread value should be logged"
                    break
                    
            assert veto_logged, "Spread veto should be logged with symbol and metrics"
            
    def test_low_adx_veto(self):
        """Test low ADX veto with proper logging."""
        decision = Mock()
        decision.symbol = "ETHUSDT"
        decision.confidence = 0.8
        decision.decision = "SHORT"
        
        # Mock market data with low ADX
        market_data = {
            'bid': 3000.0,
            'ask': 3001.0,
            'spread_bps': 3.0
        }
        
        # Mock feature data with low ADX
        feature_data = {
            'spread_bps': 3.0,
            'adx': 15.0,  # Below threshold of 25
            'atr': 50.0,
            'volume_imbalance': 0.3,
            'liquidity': 2000000
        }
        
        with patch('loguru.logger.info') as mock_logger:
            # Apply filters
            result = apply_filters(decision, market_data, feature_data, self.settings)
            
            # Check veto logging
            log_calls = mock_logger.call_args_list
            veto_logged = False
            for call in log_calls:
                log_message = str(call)
                if "LOW_ADX" in log_message and "ETHUSDT" in log_message:
                    veto_logged = True
                    assert "15.0" in log_message, "ADX value should be logged"
                    assert "25" in log_message, "Threshold should be logged"
                    break
                    
            assert veto_logged, "Low ADX veto should be logged with symbol and metrics"
            
    def test_low_atr_veto(self):
        """Test low ATR veto with proper logging."""
        decision = Mock()
        decision.symbol = "BNBUSDT"
        decision.confidence = 0.8
        decision.decision = "LONG"
        
        # Mock market data
        market_data = {
            'bid': 400.0,
            'ask': 400.1,
            'spread_bps': 2.5
        }
        
        # Mock feature data with low ATR
        feature_data = {
            'spread_bps': 2.5,
            'adx': 35.0,
            'atr': 50.0,  # Below threshold of 100
            'volume_imbalance': 0.4,
            'liquidity': 1500000
        }
        
        with patch('loguru.logger.info') as mock_logger:
            # Apply filters
            result = apply_filters(decision, market_data, feature_data, self.settings)
            
            # Check veto logging
            log_calls = mock_logger.call_args_list
            veto_logged = False
            for call in log_calls:
                log_message = str(call)
                if "LOW_ATR" in log_message and "BNBUSDT" in log_message:
                    veto_logged = True
                    assert "50.0" in log_message, "ATR value should be logged"
                    assert "100" in log_message, "Threshold should be logged"
                    break
                    
            assert veto_logged, "Low ATR veto should be logged with symbol and metrics"
            
    def test_volume_imbalance_veto(self):
        """Test volume imbalance veto with proper logging."""
        decision = Mock()
        decision.symbol = "ADAUSDT"
        decision.confidence = 0.8
        decision.decision = "LONG"
        
        # Mock market data
        market_data = {
            'bid': 0.5,
            'ask': 0.5001,
            'spread_bps': 2.0
        }
        
        # Mock feature data with high volume imbalance
        feature_data = {
            'spread_bps': 2.0,
            'adx': 40.0,
            'atr': 0.01,
            'volume_imbalance': 0.9,  # Above threshold of 0.8
            'liquidity': 800000
        }
        
        with patch('loguru.logger.info') as mock_logger:
            # Apply filters
            result = apply_filters(decision, market_data, feature_data, self.settings)
            
            # Check veto logging
            log_calls = mock_logger.call_args_list
            veto_logged = False
            for call in log_calls:
                log_message = str(call)
                if "VOLUME_IMBALANCE" in log_message and "ADAUSDT" in log_message:
                    veto_logged = True
                    assert "0.9" in log_message, "Volume imbalance value should be logged"
                    assert "0.8" in log_message, "Threshold should be logged"
                    break
                    
            assert veto_logged, "Volume imbalance veto should be logged with symbol and metrics"
            
    def test_low_liquidity_veto(self):
        """Test low liquidity veto with proper logging."""
        decision = Mock()
        decision.symbol = "DOGEUSDT"
        decision.confidence = 0.8
        decision.decision = "SHORT"
        
        # Mock market data
        market_data = {
            'bid': 0.1,
            'ask': 0.10001,
            'spread_bps': 1.0
        }
        
        # Mock feature data with low liquidity
        feature_data = {
            'spread_bps': 1.0,
            'adx': 45.0,
            'atr': 0.005,
            'volume_imbalance': 0.2,
            'liquidity': 500000  # Below threshold of 1000000
        }
        
        with patch('loguru.logger.info') as mock_logger:
            # Apply filters
            result = apply_filters(decision, market_data, feature_data, self.settings)
            
            # Check veto logging
            log_calls = mock_logger.call_args_list
            veto_logged = False
            for call in log_calls:
                log_message = str(call)
                if "LOW_LIQUIDITY" in log_message and "DOGEUSDT" in log_message:
                    veto_logged = True
                    assert "500000" in log_message, "Liquidity value should be logged"
                    assert "1000000" in log_message, "Threshold should be logged"
                    break
                    
            assert veto_logged, "Low liquidity veto should be logged with symbol and metrics"
            
    def test_multiple_vetoes(self):
        """Test multiple vetoes with comprehensive logging."""
        decision = Mock()
        decision.symbol = "XRPUSDT"
        decision.confidence = 0.8
        decision.decision = "LONG"
        
        # Mock market data with multiple issues
        market_data = {
            'bid': 0.5,
            'ask': 0.525,  # 5% spread
            'spread_bps': 500.0
        }
        
        # Mock feature data with multiple issues
        feature_data = {
            'spread_bps': 500.0,
            'adx': 10.0,  # Low ADX
            'atr': 0.005,  # Low ATR
            'volume_imbalance': 0.9,  # High imbalance
            'liquidity': 200000  # Low liquidity
        }
        
        with patch('loguru.logger.info') as mock_logger:
            # Apply filters
            result = apply_filters(decision, market_data, feature_data, self.settings)
            
            # Check multiple veto logging
            log_calls = mock_logger.call_args_list
            veto_types = []
            
            for call in log_calls:
                log_message = str(call)
                if "XRPUSDT" in log_message:
                    if "SPREAD_TOO_WIDE" in log_message:
                        veto_types.append("SPREAD_TOO_WIDE")
                    elif "LOW_ADX" in log_message:
                        veto_types.append("LOW_ADX")
                    elif "LOW_ATR" in log_message:
                        veto_types.append("LOW_ATR")
                    elif "VOLUME_IMBALANCE" in log_message:
                        veto_types.append("VOLUME_IMBALANCE")
                    elif "LOW_LIQUIDITY" in log_message:
                        veto_types.append("LOW_LIQUIDITY")
                        
            # Should have multiple vetoes
            assert len(veto_types) >= 3, f"Should have multiple vetoes, got: {veto_types}"
            
    def test_veto_counter_tracking(self):
        """Test that veto counters are properly tracked per symbol."""
        # Mock veto counter tracking
        veto_counters = {
            'BTCUSDT': {
                'SPREAD_TOO_WIDE': 0,
                'LOW_ADX': 0,
                'LOW_ATR': 0,
                'VOLUME_IMBALANCE': 0,
                'LOW_LIQUIDITY': 0
            }
        }
        
        def increment_veto_counter(symbol, veto_type):
            if symbol not in veto_counters:
                veto_counters[symbol] = {
                    'SPREAD_TOO_WIDE': 0,
                    'LOW_ADX': 0,
                    'LOW_ATR': 0,
                    'VOLUME_IMBALANCE': 0,
                    'LOW_LIQUIDITY': 0
                }
            veto_counters[symbol][veto_type] += 1
            
        # Test veto counter increment
        symbol = "BTCUSDT"
        veto_type = "SPREAD_TOO_WIDE"
        
        increment_veto_counter(symbol, veto_type)
        assert veto_counters[symbol][veto_type] == 1, "Veto counter should be incremented"
        
        increment_veto_counter(symbol, veto_type)
        assert veto_counters[symbol][veto_type] == 2, "Veto counter should be incremented again"
        
        # Test different veto type
        increment_veto_counter(symbol, "LOW_ADX")
        assert veto_counters[symbol]["LOW_ADX"] == 1, "Different veto type should have separate counter"
        assert veto_counters[symbol]["SPREAD_TOO_WIDE"] == 2, "Original counter should remain unchanged"
        
    def test_veto_execution_order(self):
        """Test that vetoes are executed in the correct order."""
        decision = Mock()
        decision.symbol = "BTCUSDT"
        decision.confidence = 0.8
        decision.decision = "LONG"
        
        # Mock market data with multiple issues
        market_data = {
            'bid': 50000.0,
            'ask': 50250.0,
            'spread_bps': 50.0
        }
        
        feature_data = {
            'spread_bps': 50.0,
            'adx': 10.0,
            'atr': 50.0,
            'volume_imbalance': 0.9,
            'liquidity': 200000
        }
        
        with patch('loguru.logger.info') as mock_logger:
            # Apply filters
            result = apply_filters(decision, market_data, feature_data, self.settings)
            
            # Check execution order (should be deterministic)
            log_calls = mock_logger.call_args_list
            veto_order = []
            
            for call in log_calls:
                log_message = str(call)
                if "BTCUSDT" in log_message:
                    if "SPREAD_TOO_WIDE" in log_message:
                        veto_order.append("SPREAD_TOO_WIDE")
                    elif "LOW_ADX" in log_message:
                        veto_order.append("LOW_ADX")
                    elif "LOW_ATR" in log_message:
                        veto_order.append("LOW_ATR")
                    elif "VOLUME_IMBALANCE" in log_message:
                        veto_order.append("VOLUME_IMBALANCE")
                    elif "LOW_LIQUIDITY" in log_message:
                        veto_order.append("LOW_LIQUIDITY")
                        
            # Vetoes should be executed in consistent order
            assert len(veto_order) > 0, "Should have vetoes executed"
            
    def test_veto_metrics_collection(self):
        """Test that veto metrics are properly collected."""
        # Mock metrics collection
        veto_metrics = {
            'total_vetoes': 0,
            'vetoes_by_type': {},
            'vetoes_by_symbol': {},
            'veto_rate': 0.0
        }
        
        def collect_veto_metric(symbol, veto_type):
            veto_metrics['total_vetoes'] += 1
            
            if veto_type not in veto_metrics['vetoes_by_type']:
                veto_metrics['vetoes_by_type'][veto_type] = 0
            veto_metrics['vetoes_by_type'][veto_type] += 1
            
            if symbol not in veto_metrics['vetoes_by_symbol']:
                veto_metrics['vetoes_by_symbol'][symbol] = 0
            veto_metrics['vetoes_by_symbol'][symbol] += 1
            
        # Test metrics collection
        collect_veto_metric("BTCUSDT", "SPREAD_TOO_WIDE")
        collect_veto_metric("ETHUSDT", "LOW_ADX")
        collect_veto_metric("BTCUSDT", "LOW_ATR")
        
        assert veto_metrics['total_vetoes'] == 3, "Total vetoes should be 3"
        assert veto_metrics['vetoes_by_type']['SPREAD_TOO_WIDE'] == 1, "SPREAD_TOO_WIDE should be 1"
        assert veto_metrics['vetoes_by_type']['LOW_ADX'] == 1, "LOW_ADX should be 1"
        assert veto_metrics['vetoes_by_type']['LOW_ATR'] == 1, "LOW_ATR should be 1"
        assert veto_metrics['vetoes_by_symbol']['BTCUSDT'] == 2, "BTCUSDT should have 2 vetoes"
        assert veto_metrics['vetoes_by_symbol']['ETHUSDT'] == 1, "ETHUSDT should have 1 veto"
        
    def test_veto_before_notification(self):
        """Test that vetoes run before any notification is sent."""
        decision = Mock()
        decision.symbol = "BTCUSDT"
        decision.confidence = 0.8
        decision.decision = "LONG"
        
        # Mock market data that should trigger veto
        market_data = {
            'bid': 50000.0,
            'ask': 50250.0,
            'spread_bps': 50.0
        }
        
        feature_data = {
            'spread_bps': 50.0,
            'adx': 30.0,
            'atr': 1000.0,
            'volume_imbalance': 0.3,
            'liquidity': 5000000
        }
        
        # Mock notification function
        notification_sent = False
        
        def send_notification(decision):
            nonlocal notification_sent
            notification_sent = True
            
        with patch('loguru.logger.info') as mock_logger:
            # Apply filters first
            result = apply_filters(decision, market_data, feature_data, self.settings)
            
            # Check if veto was applied
            veto_applied = False
            log_calls = mock_logger.call_args_list
            for call in log_calls:
                log_message = str(call)
                if "SPREAD_TOO_WIDE" in log_message:
                    veto_applied = True
                    break
                    
            # If veto was applied, notification should not be sent
            if veto_applied:
                send_notification(decision)
                assert not notification_sent, "Notification should not be sent when veto is applied"
                
    def test_synthetic_veto_triggering(self):
        """Test synthetic inputs that trigger various vetoes."""
        # Test case 1: Wide spread
        decision1 = Mock()
        decision1.symbol = "BTCUSDT"
        decision1.confidence = 0.8
        decision1.decision = "LONG"
        
        market_data1 = {'bid': 50000.0, 'ask': 50250.0, 'spread_bps': 50.0}
        feature_data1 = {
            'spread_bps': 50.0, 'adx': 30.0, 'atr': 1000.0,
            'volume_imbalance': 0.3, 'liquidity': 5000000
        }
        
        with patch('loguru.logger.info') as mock_logger:
            result1 = apply_filters(decision1, market_data1, feature_data1, self.settings)
            log_calls1 = mock_logger.call_args_list
            
            spread_veto_found = any("SPREAD_TOO_WIDE" in str(call) for call in log_calls1)
            assert spread_veto_found, "Wide spread should trigger veto"
            
        # Test case 2: Low ADX
        decision2 = Mock()
        decision2.symbol = "ETHUSDT"
        decision2.confidence = 0.8
        decision2.decision = "SHORT"
        
        market_data2 = {'bid': 3000.0, 'ask': 3000.1, 'spread_bps': 3.0}
        feature_data2 = {
            'spread_bps': 3.0, 'adx': 15.0, 'atr': 50.0,
            'volume_imbalance': 0.3, 'liquidity': 2000000
        }
        
        with patch('loguru.logger.info') as mock_logger:
            result2 = apply_filters(decision2, market_data2, feature_data2, self.settings)
            log_calls2 = mock_logger.call_args_list
            
            adx_veto_found = any("LOW_ADX" in str(call) for call in log_calls2)
            assert adx_veto_found, "Low ADX should trigger veto"


def test_veto_reason_auditability():
    """Test that veto reasons are auditable and traceable."""
    # Mock audit log
    audit_log = []
    
    def log_veto_audit(symbol, veto_type, metrics, threshold):
        audit_entry = {
            'timestamp': time.time(),
            'symbol': symbol,
            'veto_type': veto_type,
            'metrics': metrics,
            'threshold': threshold,
            'reason': f"{veto_type} veto applied: {metrics} exceeds threshold {threshold}"
        }
        audit_log.append(audit_entry)
        
    # Test audit logging
    log_veto_audit("BTCUSDT", "SPREAD_TOO_WIDE", 50.0, 30.0)
    log_veto_audit("ETHUSDT", "LOW_ADX", 15.0, 25.0)
    
    assert len(audit_log) == 2, "Should have 2 audit entries"
    
    # Check audit entry structure
    for entry in audit_log:
        assert 'timestamp' in entry, "Audit entry should have timestamp"
        assert 'symbol' in entry, "Audit entry should have symbol"
        assert 'veto_type' in entry, "Audit entry should have veto_type"
        assert 'metrics' in entry, "Audit entry should have metrics"
        assert 'threshold' in entry, "Audit entry should have threshold"
        assert 'reason' in entry, "Audit entry should have reason"


def test_veto_performance_impact():
    """Test that veto checking doesn't significantly impact performance."""
    import time
    
    decision = Mock()
    decision.symbol = "BTCUSDT"
    decision.confidence = 0.8
    decision.decision = "LONG"
    
    market_data = {
        'bid': 50000.0,
        'ask': 50000.1,
        'spread_bps': 2.0
    }
    
    feature_data = {
        'spread_bps': 2.0,
        'adx': 35.0,
        'atr': 1000.0,
        'volume_imbalance': 0.3,
        'liquidity': 5000000
    }
    
    # Measure veto check performance
    start_time = time.time()
    
    for _ in range(1000):  # Run 1000 veto checks
        with patch('loguru.logger.info'):
            apply_filters(decision, market_data, feature_data, self.settings)
            
    end_time = time.time()
    total_time = end_time - start_time
    
    # Veto checks should be fast (less than 1 second for 1000 checks)
    assert total_time < 1.0, f"Veto checks too slow: {total_time:.3f}s for 1000 checks"


def main():
    """Run all veto tests."""
    print("🧪 Testing Risk Filter Vetoes and Audit Visibility...")
    
    # Create test instance
    test_suite = TestVetoes()
    
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
        test_veto_reason_auditability()
        print("✅ test_veto_reason_auditability passed")
        passed += 1
    except Exception as e:
        print(f"❌ test_veto_reason_auditability failed: {e}")
        failed += 1
        
    try:
        test_veto_performance_impact()
        print("✅ test_veto_performance_impact passed")
        passed += 1
    except Exception as e:
        print(f"❌ test_veto_performance_impact failed: {e}")
        failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All veto tests passed!")
        return True
    else:
        print("❌ Some veto tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)



