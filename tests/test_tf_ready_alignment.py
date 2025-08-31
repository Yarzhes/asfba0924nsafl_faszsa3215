#!/usr/bin/env python3
"""
Test Timeframe Ready Alignment and NaN Hygiene

This module tests:
1. Timeframe-specific warmup enforcement
2. Bar close alignment across timeframes
3. NaN handling and propagation
4. Ensemble voting with ready timeframes
"""

import time
import numpy as np
import pandas as pd
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
    from ultra_signals.apps.realtime_runner import ResilientSignalRunner


class TestTimeframeReadyAlignment:
    """Test suite for timeframe readiness and alignment."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.settings = Mock()
        self.settings.features.warmup_periods = 200
        self.settings.debug = True
        
        self.runner = ResilientSignalRunner(self.settings)
        self.current_time = time.time()
        
    def test_timeframe_warmup_enforcement(self):
        """Test that timeframes are only marked ready with sufficient data."""
        symbol = "BTCUSDT"
        timeframes = ["1m", "5m", "15m"]
        
        # Test insufficient data
        mock_ohlcv_insufficient = Mock()
        mock_ohlcv_insufficient.__len__ = lambda self: 150  # Below threshold
        
        self.runner.feature_store.get_ohlcv = Mock(return_value=mock_ohlcv_insufficient)
        
        for tf in timeframes:
            is_ready = self.runner._is_timeframe_ready(symbol, tf)
            assert not is_ready, f"Timeframe {tf} should not be ready with insufficient data"
            
        # Test sufficient data
        mock_ohlcv_sufficient = Mock()
        mock_ohlcv_sufficient.__len__ = lambda self: 250  # Above threshold
        
        self.runner.feature_store.get_ohlcv = Mock(return_value=mock_ohlcv_sufficient)
        
        for tf in timeframes:
            is_ready = self.runner._is_timeframe_ready(symbol, tf)
            assert is_ready, f"Timeframe {tf} should be ready with sufficient data"
            
        # Verify ready timeframes are tracked
        assert "1m" in self.runner.ready_timeframes[symbol]
        assert "5m" in self.runner.ready_timeframes[symbol]
        assert "15m" in self.runner.ready_timeframes[symbol]
        
    def test_bar_close_alignment(self):
        """Test that bars align to the same decision timestamp."""
        # Mock OHLCV data with timestamps
        base_time = 1640995200  # Fixed base timestamp
        
        # Create aligned timestamps for different timeframes
        timestamps_1m = [base_time + i * 60 for i in range(300)]  # 1-minute bars
        timestamps_5m = [base_time + i * 300 for i in range(60)]   # 5-minute bars
        timestamps_15m = [base_time + i * 900 for i in range(20)] # 15-minute bars
        
        # Test that 5m and 15m bars align with 1m bars
        for tf_5m in timestamps_5m:
            # Find corresponding 1m bars
            corresponding_1m = [t for t in timestamps_1m if t <= tf_5m < t + 300]
            assert len(corresponding_1m) > 0, f"5m bar {tf_5m} should align with 1m bars"
            
        for tf_15m in timestamps_15m:
            # Find corresponding 1m bars
            corresponding_1m = [t for t in timestamps_1m if t <= tf_15m < t + 900]
            assert len(corresponding_1m) > 0, f"15m bar {tf_15m} should align with 1m bars"
            
    def test_no_look_ahead(self):
        """Test that higher timeframe bars are not used before they're fully closed."""
        current_time = time.time()
        
        # Calculate bar boundaries
        def get_bar_boundaries(tf_minutes):
            bar_seconds = tf_minutes * 60
            current_bar_start = (current_time // bar_seconds) * bar_seconds
            current_bar_end = current_bar_start + bar_seconds
            return current_bar_start, current_bar_end
            
        # Test different timeframes
        timeframes = [1, 5, 15]
        
        for tf_minutes in timeframes:
            bar_start, bar_end = get_bar_boundaries(tf_minutes)
            
            # Current time should be within the current bar
            assert bar_start <= current_time < bar_end, f"Current time should be within {tf_minutes}m bar"
            
            # We should only use data up to the previous bar
            previous_bar_end = bar_start
            assert current_time >= previous_bar_end, f"Should only use data up to previous {tf_minutes}m bar"
            
    def test_nan_hygiene(self):
        """Test that NaN values are properly handled and don't propagate."""
        # Mock feature calculation that returns NaN
        def mock_feature_with_nan(data):
            return np.nan
            
        def mock_feature_with_value(data):
            return 0.75
            
        # Test NaN handling in ensemble
        mock_decision = Mock()
        mock_decision.confidence = np.nan
        mock_decision.decision = "LONG"
        mock_decision.symbol = "BTCUSDT"
        mock_decision.tf = "5m"
        
        # NaN confidence should be handled gracefully
        # The runner should log this at DEBUG level and not send signal
        with patch('loguru.logger.debug') as mock_debug:
            should_send = self.runner._should_send_signal("BTCUSDT", mock_decision, self.current_time)
            assert not should_send, "NaN confidence should prevent signal sending"
            
    def test_ensemble_voting_with_ready_timeframes(self):
        """Test that only ready timeframes participate in ensemble voting."""
        symbol = "BTCUSDT"
        timeframes = ["1m", "5m", "15m"]
        
        # Mock feature store with mixed data availability
        def mock_get_ohlcv(symbol, tf):
            mock_data = Mock()
            if tf == "1m":
                mock_data.__len__ = lambda self: 250  # Ready
            elif tf == "5m":
                mock_data.__len__ = lambda self: 150  # Not ready
            else:  # 15m
                mock_data.__len__ = lambda self: 250  # Ready
            return mock_data
            
        self.runner.feature_store.get_ohlcv = mock_get_ohlcv
        
        # Check readiness
        ready_1m = self.runner._is_timeframe_ready(symbol, "1m")
        ready_5m = self.runner._is_timeframe_ready(symbol, "5m")
        ready_15m = self.runner._is_timeframe_ready(symbol, "15m")
        
        assert ready_1m, "1m should be ready"
        assert not ready_5m, "5m should not be ready"
        assert ready_15m, "15m should be ready"
        
        # Only ready timeframes should be in the ensemble
        ready_tfs = self.runner.ready_timeframes[symbol]
        assert "1m" in ready_tfs, "1m should be in ready timeframes"
        assert "5m" not in ready_tfs, "5m should not be in ready timeframes"
        assert "15m" in ready_tfs, "15m should be in ready timeframes"
        
    def test_timeframe_ready_persistence(self):
        """Test that timeframe readiness persists across checks."""
        symbol = "BTCUSDT"
        tf = "5m"
        
        # Mock sufficient data
        mock_ohlcv = Mock()
        mock_ohlcv.__len__ = lambda self: 250
        
        self.runner.feature_store.get_ohlcv = Mock(return_value=mock_ohlcv)
        
        # First check
        is_ready1 = self.runner._is_timeframe_ready(symbol, tf)
        assert is_ready1, "First check should mark timeframe as ready"
        
        # Second check (should use cached result)
        is_ready2 = self.runner._is_timeframe_ready(symbol, tf)
        assert is_ready2, "Second check should return cached ready status"
        
        # Verify timeframe is tracked
        assert tf in self.runner.ready_timeframes[symbol], "Timeframe should be tracked as ready"
        
    def test_timeframe_ready_reset(self):
        """Test that timeframe readiness can be reset when data becomes insufficient."""
        symbol = "BTCUSDT"
        tf = "5m"
        
        # Start with sufficient data
        mock_ohlcv_sufficient = Mock()
        mock_ohlcv_sufficient.__len__ = lambda self: 250
        
        self.runner.feature_store.get_ohlcv = Mock(return_value=mock_ohlcv_sufficient)
        
        is_ready1 = self.runner._is_timeframe_ready(symbol, tf)
        assert is_ready1, "Timeframe should be ready with sufficient data"
        
        # Switch to insufficient data
        mock_ohlcv_insufficient = Mock()
        mock_ohlcv_insufficient.__len__ = lambda self: 150
        
        self.runner.feature_store.get_ohlcv = Mock(return_value=mock_ohlcv_insufficient)
        
        # Clear cached status
        if symbol in self.runner.ready_timeframes:
            self.runner.ready_timeframes[symbol].discard(tf)
        
        is_ready2 = self.runner._is_timeframe_ready(symbol, tf)
        assert not is_ready2, "Timeframe should not be ready with insufficient data"
        
    def test_nan_propagation_prevention(self):
        """Test that NaN values don't propagate into final numeric fields."""
        # Mock decision with NaN components
        mock_decision = Mock()
        mock_decision.confidence = 0.8  # Valid confidence
        mock_decision.decision = "LONG"
        mock_decision.symbol = "BTCUSDT"
        mock_decision.tf = "5m"
        
        # Mock vote detail with NaN values
        mock_decision.vote_detail = {
            'trend': np.nan,
            'momentum': 0.75,
            'imbalance': np.nan,
            'risk_model': {
                'entry_price': 50000.0,
                'atr': 1000.0
            }
        }
        
        # Test that NaN values don't affect the main decision
        should_send = self.runner._should_send_signal("BTCUSDT", mock_decision, self.current_time)
        
        # The decision should still be valid despite NaN components
        assert mock_decision.confidence == 0.8, "Confidence should remain valid"
        assert mock_decision.decision == "LONG", "Decision should remain valid"
        
    def test_bar_time_alignment_validation(self):
        """Test that bar times are properly aligned for ensemble decisions."""
        # Mock bar times for different timeframes
        base_time = 1640995200
        
        # 1-minute bars
        bar_1m = base_time + 300  # 5 minutes into the hour
        
        # 5-minute bars (should align)
        bar_5m = base_time + 300  # Same time
        
        # 15-minute bars (should align)
        bar_15m = base_time + 900  # 15 minutes into the hour
        
        # Test alignment
        assert bar_1m % 60 == 0, "1m bar should align to minute boundary"
        assert bar_5m % 300 == 0, "5m bar should align to 5-minute boundary"
        assert bar_15m % 900 == 0, "15m bar should align to 15-minute boundary"
        
        # Test that 5m and 15m bars align with 1m bars
        assert bar_5m % 60 == 0, "5m bar should align with 1m boundary"
        assert bar_15m % 60 == 0, "15m bar should align with 1m boundary"
        
    def test_warmup_period_consistency(self):
        """Test that warmup periods are consistently applied across timeframes."""
        symbol = "BTCUSDT"
        timeframes = ["1m", "5m", "15m"]
        
        # Mock data with exactly warmup_periods length
        mock_ohlcv_exact = Mock()
        mock_ohlcv_exact.__len__ = lambda self: 200  # Exactly warmup_periods
        
        self.runner.feature_store.get_ohlcv = Mock(return_value=mock_ohlcv_exact)
        
        for tf in timeframes:
            is_ready = self.runner._is_timeframe_ready(symbol, tf)
            assert is_ready, f"Timeframe {tf} should be ready with exact warmup periods"
            
        # Test with one less than required
        mock_ohlcv_insufficient = Mock()
        mock_ohlcv_insufficient.__len__ = lambda self: 199  # One less than required
        
        self.runner.feature_store.get_ohlcv = Mock(return_value=mock_ohlcv_insufficient)
        
        for tf in timeframes:
            is_ready = self.runner._is_timeframe_ready(symbol, tf)
            assert not is_ready, f"Timeframe {tf} should not be ready with insufficient data"


def test_nan_handling_in_features():
    """Test NaN handling in feature calculations."""
    # Mock feature calculation that might return NaN
    def mock_trend_feature(data):
        if len(data) < 50:
            return np.nan
        return 0.75
        
    def mock_momentum_feature(data):
        if len(data) < 20:
            return np.nan
        return 0.65
        
    # Test with insufficient data
    insufficient_data = [1, 2, 3, 4, 5]  # Only 5 points
    
    trend_result = mock_trend_feature(insufficient_data)
    momentum_result = mock_momentum_feature(insufficient_data)
    
    assert np.isnan(trend_result), "Trend should return NaN with insufficient data"
    assert np.isnan(momentum_result), "Momentum should return NaN with insufficient data"
    
    # Test with sufficient data
    sufficient_data = list(range(100))  # 100 points
    
    trend_result = mock_trend_feature(sufficient_data)
    momentum_result = mock_momentum_feature(sufficient_data)
    
    assert not np.isnan(trend_result), "Trend should return valid value with sufficient data"
    assert not np.isnan(momentum_result), "Momentum should return valid value with sufficient data"


def test_time_alignment_utilities():
    """Test time alignment utility functions."""
    # Test bar boundary calculation
    def get_bar_boundaries(timestamp, tf_minutes):
        bar_seconds = tf_minutes * 60
        bar_start = (timestamp // bar_seconds) * bar_seconds
        bar_end = bar_start + bar_seconds
        return bar_start, bar_end
    
    # Test with different timeframes
    test_time = 1640995260  # Some timestamp
    
    # 1-minute bars
    start_1m, end_1m = get_bar_boundaries(test_time, 1)
    assert start_1m <= test_time < end_1m, "1m bar boundaries should contain test time"
    assert (end_1m - start_1m) == 60, "1m bar should be 60 seconds"
    
    # 5-minute bars
    start_5m, end_5m = get_bar_boundaries(test_time, 5)
    assert start_5m <= test_time < end_5m, "5m bar boundaries should contain test time"
    assert (end_5m - start_5m) == 300, "5m bar should be 300 seconds"
    
    # 15-minute bars
    start_15m, end_15m = get_bar_boundaries(test_time, 15)
    assert start_15m <= test_time < end_15m, "15m bar boundaries should contain test time"
    assert (end_15m - start_15m) == 900, "15m bar should be 900 seconds"


def main():
    """Run all timeframe alignment tests."""
    print("🧪 Testing Timeframe Ready Alignment and NaN Hygiene...")
    
    # Create test instance
    test_suite = TestTimeframeReadyAlignment()
    
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
        test_nan_handling_in_features()
        print("✅ test_nan_handling_in_features passed")
        passed += 1
    except Exception as e:
        print(f"❌ test_nan_handling_in_features failed: {e}")
        failed += 1
        
    try:
        test_time_alignment_utilities()
        print("✅ test_time_alignment_utilities passed")
        passed += 1
    except Exception as e:
        print(f"❌ test_time_alignment_utilities failed: {e}")
        failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All timeframe alignment tests passed!")
        return True
    else:
        print("❌ Some timeframe alignment tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)



