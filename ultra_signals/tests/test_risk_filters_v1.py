"""
Tests for v1 Risk Filters
"""
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from ultra_signals.engine.risk_filters import apply_filters
from ultra_signals.core.custom_types import Signal, SignalType, PortfolioState
from unittest.mock import MagicMock

@pytest.fixture
def mock_signal() -> Signal:
    """A mock Signal object for testing."""
    return Signal(
        symbol="BTCUSDT",
        timeframe="5m",
        decision="LONG",
        signal_type=SignalType.TREND_FOLLOWING,
        price=50000,
        stop_loss=49500,
        take_profit_1=51000,
        score=0.8,
        features={},
    )

@pytest.fixture
def mock_settings() -> dict:
    """A mock settings dictionary for testing."""
    return {
        "features": {"warmup_periods": 50},
        "filters": {"max_spread_pct": 0.002}
    }

def test_apply_filters_pass():
    """Test apply_filters passes when conditions are met"""
    signal = Signal(
        symbol="BTCUSDT",
        timeframe="5m",
        decision="LONG",
        signal_type=SignalType.BREAKOUT,
        price=50000,
        stop_loss=49500,
        take_profit_1=51000,
        score=0.8
    )
    settings = {"features": {"warmup_periods": 50}, "filters": {"max_spread_pct": 0.002}}
    fs = MagicMock()
    fs.get_warmup_status.return_value = 60
    fs.get_book_ticker.return_value = (100.0, 100.1, 0.1, 10)
    
    result = apply_filters(signal, fs, settings)
    assert result.passed is True

def test_apply_filters_fail_warmup():
    """Test apply_filters fails when warmup not met"""
    signal = Signal(
        symbol="BTCUSDT",
        timeframe="5m",
        decision="LONG",
        signal_type=SignalType.BREAKOUT,
        price=50000,
        stop_loss=49500,
        take_profit_1=51000,
        score=0.8
    )
    settings = {"features": {"warmup_periods": 50}, "filters": {"max_spread_pct": 0.002}}
    fs = MagicMock()
    fs.get_warmup_status.return_value = 40
    fs.get_book_ticker.return_value = (100.0, 100.1, 0.1, 10)
    
    result = apply_filters(signal, fs, settings)
    assert result.passed is False

def test_apply_filters_fail_spread():
    """Test apply_filters fails when spread too high"""
    signal = Signal(
        symbol="BTCUSDT",
        timeframe="5m",
        decision="LONG",
        signal_type=SignalType.BREAKOUT,
        price=50000,
        stop_loss=49500,
        take_profit_1=51000,
        score=0.8
    )
    settings = {"features": {"warmup_periods": 50}, "filters": {"max_spread_pct": 0.002}}
    fs = MagicMock()
    fs.get_warmup_status.return_value = 60
    fs.get_book_ticker.return_value = (100.0, 101.0, 1.0, 10)
    
    result = apply_filters(signal, fs, settings)
    assert result.passed is False


def test_funding_time_logic():
    """ Test the funding time logic directly """
    from ultra_signals.core.timeutils import is_funding_imminent
    
    # Test cases for is_funding_imminent
    # Expected to be inside the 10-minute window (fail)
    assert is_funding_imminent(pd.to_datetime("2023-01-01 07:55:00").timestamp() * 1000, 10) is True
    assert is_funding_imminent(pd.to_datetime("2023-01-01 08:05:00").timestamp() * 1000, 10) is True
    assert is_funding_imminent(pd.to_datetime("2023-01-01 23:58:00").timestamp() * 1000, 10) is True # Next day check
    
    # Expected to be outside the 10-minute window (pass)
    assert is_funding_imminent(pd.to_datetime("2023-01-01 07:45:00").timestamp() * 1000, 10) is False
    assert is_funding_imminent(pd.to_datetime("2023-01-01 08:15:00").timestamp() * 1000, 10) is False