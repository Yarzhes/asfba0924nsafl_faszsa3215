"""
Integration smoke test for realtime runner with sniper mode.
"""

import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from ultra_signals.core.config import Settings
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import Signal, SignalType
from ultra_signals.engine.sniper_counters import reset_sniper_counters


class MockWebSocketClient:
    """Mock WebSocket client for testing."""
    
    def __init__(self):
        self.is_connected = True
        self._signal_queue = asyncio.Queue()
    
    async def connect(self):
        pass
    
    async def disconnect(self):
        pass
    
    async def get_signal(self):
        """Return a test signal."""
        return await self._signal_queue.get()
    
    def add_test_signal(self, signal):
        """Add a signal for testing."""
        try:
            self._signal_queue.put_nowait(signal)
        except asyncio.QueueFull:
            pass


class MockTelegramTransport:
    """Mock Telegram transport for testing."""
    
    def __init__(self):
        self.sent_messages = []
    
    async def send_message(self, message, **kwargs):
        self.sent_messages.append(message)
        return True


def make_test_signal(symbol="BTCUSDT"):
    return Signal(
        symbol=symbol,
        timeframe="5m",
        decision="LONG",
        signal_type=SignalType.TREND_FOLLOWING,
        price=50000,
        stop_loss=49500,
        take_profit_1=51000,
        score=0.9,
        features={},
    )


@pytest.fixture
def mock_settings():
    """Create test settings with sniper mode enabled."""
    return {
        "features": {"warmup_periods": 20},
        "runtime": {
            "sniper_mode": {
                "enabled": True,
                "max_signals_per_hour": 2,
                "daily_signal_cap": 10,
                "mtf_confirm": False
            }
        },
        "redis": {"enabled": False},  # Use memory backend for testing
        "transport": {
            "telegram": {
                "enabled": True,
                "dry_run": True
            }
        }
    }


@pytest.fixture
def mock_feature_store():
    """Create mock feature store."""
    store = MagicMock(spec=FeatureStore)
    store.get_warmup_status.return_value = 100
    store.get_book_ticker.return_value = (100.0, 100.1, 0.1, 10)
    store.current_ts_ms.return_value = int(time.time() * 1000)
    return store


@pytest.mark.asyncio
async def test_sniper_integration_hourly_cap_removed(mock_settings, mock_feature_store):
    """Hourly cap has been removed: all signals should pass (no SNIPER_HOURLY_CAP blocks)."""
    from ultra_signals.engine.risk_filters import apply_filters
    from ultra_signals.live.metrics import Metrics

    reset_sniper_counters()
    metrics = Metrics()
    processed_signals, blocked_signals = [], []

    for i in range(4):  # previously > hourly cap; now all should pass
        signal = make_test_signal(f"BTC{i}USDT")
        res = apply_filters(signal, mock_feature_store, mock_settings)
        if not res.passed:
            blocked_signals.append((signal, res.reason))
            if 'SNIPER' in res.reason:
                metrics.inc_sniper_rejection(res.reason)
        else:
            processed_signals.append(signal)

    assert len(processed_signals) == 4, f"Expected all signals to pass, got {len(processed_signals)}"
    assert len(blocked_signals) == 0, f"Expected no blocked signals, got {len(blocked_signals)}"
    snap = metrics.snapshot()
    # Hourly counter should remain zero since no hourly cap enforced
    assert snap['counters']['sniper_hourly_cap'] == 0


@pytest.mark.asyncio
async def test_sniper_integration_mtf_confirm(mock_feature_store):
    """Test that sniper MTF confirmation is enforced."""
    from ultra_signals.engine.risk_filters import apply_filters
    
    # Reset counters
    reset_sniper_counters()
    
    settings_with_mtf = {
        "features": {"warmup_periods": 20},
        "runtime": {
            "sniper_mode": {
                "enabled": True,
                "max_signals_per_hour": 10,
                "daily_signal_cap": 10,
                "mtf_confirm": True  # Require MTF confirmation
            }
        },
        "confluence": {
            "require_regime_align": True,
            "map": {"5m": "15m"}  # Map 5m to 15m for HTF check
        },
        "redis": {"enabled": False}
    }
    
    # Mock regime disagreement
    mock_feature_store.get_regime.return_value = "trend_down"  # HTF disagrees with LONG signal
    
    signal = make_test_signal()
    signal.decision = "LONG"  # Should be blocked by trend_down regime
    
    result = apply_filters(signal, mock_feature_store, settings_with_mtf)
    
    # Should be blocked due to MTF disagreement
    assert result.passed is False
    assert 'MTF' in result.reason or 'SNIPER_MTF' in result.reason


def test_metrics_sniper_counters():
    """Test that metrics correctly track sniper rejections."""
    from ultra_signals.live.metrics import Metrics
    
    metrics = Metrics()
    
    # Test counter increments
    metrics.inc_sniper_rejection('SNIPER_HOURLY_CAP')
    metrics.inc_sniper_rejection('SNIPER_DAILY_CAP')
    metrics.inc_sniper_rejection('SNIPER_MTF_REQUIRED')
    
    snapshot = metrics.snapshot()
    counters = snapshot['counters']
    
    assert counters['sniper_hourly_cap'] == 1
    assert counters['sniper_daily_cap'] == 1
    assert counters['sniper_mtf_required'] == 1


def test_prometheus_export_includes_sniper_metrics():
    """Test that Prometheus export includes sniper metrics."""
    from ultra_signals.live.metrics import Metrics
    
    metrics = Metrics()
    
    # Add some sniper rejections
    metrics.inc_sniper_rejection('SNIPER_HOURLY_CAP')
    metrics.inc_sniper_rejection('SNIPER_DAILY_CAP')
    
    prom_output = metrics.to_prometheus()
    
    # Check that sniper metrics are included
    assert 'sniper_hourly_cap_total' in prom_output
    assert 'sniper_daily_cap_total' in prom_output
    assert 'sniper_mtf_required_total' in prom_output
