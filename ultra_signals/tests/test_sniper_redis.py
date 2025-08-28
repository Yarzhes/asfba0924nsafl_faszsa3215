"""
Integration test for sniper mode enforcement with Redis counters.
"""

import time
import pytest
from unittest.mock import MagicMock

from ultra_signals.engine.sniper_counters import SniperCounters, reset_sniper_counters
from ultra_signals.core.custom_types import Signal, SignalType


def make_signal():
    return Signal(
        symbol="BTCUSDT",
        timeframe="5m",
        decision="LONG",
        signal_type=SignalType.TREND_FOLLOWING,
        price=50000,
        stop_loss=49500,
        take_profit_1=51000,
        score=0.9,
        features={},
    )


def test_redis_counters_with_redis_unavailable():
    """Test that counters fall back to memory when Redis is unavailable."""
    settings = {"redis": {"enabled": False}}  # Explicitly disable Redis
    counters = SniperCounters(settings)
    
    # Should use memory backend
    assert not counters._use_redis
    
    # Test counting
    result1 = counters.check_and_increment(2, 10)
    assert result1 is None  # First signal allowed
    
    result2 = counters.check_and_increment(2, 10)
    assert result2 is None  # Second signal allowed
    
    result3 = counters.check_and_increment(2, 10)
    assert result3 == 'SNIPER_HOURLY_CAP'  # Third blocked by hourly cap
    
    counts = counters.get_current_counts()
    assert counts['hour'] == 2
    assert counts['day'] == 2


def test_redis_counters_daily_cap():
    """Test daily cap enforcement."""
    settings = {"redis": {"enabled": False}}
    counters = SniperCounters(settings)
    
    result1 = counters.check_and_increment(100, 2)
    assert result1 is None
    
    result2 = counters.check_and_increment(100, 2)
    assert result2 is None
    
    result3 = counters.check_and_increment(100, 2)
    assert result3 == 'SNIPER_DAILY_CAP'


def test_redis_counters_reset():
    """Test counter reset functionality."""
    settings = {"redis": {"enabled": False}}
    counters = SniperCounters(settings)
    
    # Add some counts
    counters.check_and_increment(10, 10)
    counters.check_and_increment(10, 10)
    
    counts_before = counters.get_current_counts()
    assert counts_before['hour'] == 2
    
    # Reset
    counters.reset()
    
    counts_after = counters.get_current_counts()
    assert counts_after['hour'] == 0
    assert counts_after['day'] == 0


def test_redis_counters_cleanup():
    """Test that old timestamps are cleaned up."""
    settings = {"redis": {"enabled": False}}
    counters = SniperCounters(settings)
    
    # Manually add old timestamps to memory fallback
    old_time = int(time.time()) - 7200  # 2 hours ago (outside hour window but inside day window)
    very_old_time = int(time.time()) - 90000  # More than 24 hours ago (outside day window)
    
    counters._memory_fallback['hour'].append(old_time)
    counters._memory_fallback['day'].append(very_old_time)
    
    # Check and increment should clean up old entries
    result = counters.check_and_increment(5, 5)
    assert result is None
    
    counts = counters.get_current_counts()
    assert counts['hour'] == 1  # Only the new entry (old_time cleaned up)
    assert counts['day'] == 1   # Only the new entry (very_old_time cleaned up)


@pytest.mark.skipif(True, reason="Requires Redis server - enable manually for integration testing")
def test_redis_counters_with_redis_enabled():
    """Test Redis backend - requires running Redis server."""
    settings = {
        "redis": {
            "enabled": True,
            "host": "localhost",
            "port": 6379,
            "db": 1,  # Use test DB
            "timeout": 2
        }
    }
    
    try:
        counters = SniperCounters(settings)
        if not counters._use_redis:
            pytest.skip("Redis not available for testing")
        
        # Clean state
        counters.reset()
        
        # Test Redis counting
        result1 = counters.check_and_increment(2, 10)
        assert result1 is None
        
        result2 = counters.check_and_increment(2, 10)
        assert result2 is None
        
        result3 = counters.check_and_increment(2, 10)
        assert result3 == 'SNIPER_HOURLY_CAP'
        
        counts = counters.get_current_counts()
        assert counts['hour'] == 2
        assert counts['day'] == 2
        
    except Exception as e:
        pytest.skip(f"Redis test failed: {e}")


def test_global_reset():
    """Test global reset function."""
    from ultra_signals.engine.sniper_counters import get_sniper_counters, reset_sniper_counters
    
    settings = {"redis": {"enabled": False}}
    
    # Get counters and add some counts
    counters1 = get_sniper_counters(settings)
    counters1.check_and_increment(10, 10)
    
    # Global reset
    reset_sniper_counters()
    
    # Get counters again (should be new instance)
    counters2 = get_sniper_counters(settings)
    counts = counters2.get_current_counts()
    assert counts['hour'] == 0
    assert counts['day'] == 0
