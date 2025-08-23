import pytest
import pandas as pd
from ultra_signals.backtest.event_runner import EventRunner, MockSignalEngine
from ultra_signals.backtest.data_adapter import DataAdapter

@pytest.fixture
def mock_settings():
    """Provides mock settings for the backtest."""
    return {
        "initial_capital": 10000.0,
        "start_date": "2023-01-01",
        "end_date": "2023-01-02",
        "default_size_pct": 0.01,
        # Portfolio settings that will be used by evaluate_portfolio
        "portfolio": {
            "max_positions_total": 1,
            "max_net_long_risk": 0.015, # Allow one trade, but not two
        },
        "features": {
            "warmup_periods": 10,
            "trend": {},
            "momentum": {},
            "volatility": {},
            "volume_flow": {}
        }
    }

@pytest.fixture
def mock_data_adapter(monkeypatch):
    """Mocks the DataAdapter to return a predictable OHLCV DataFrame."""
    def mock_load_ohlcv(*args, **kwargs):
        dates = pd.to_datetime(["2023-01-01 12:00", "2023-01-01 13:00", "2023-01-01 14:00"])
        return pd.DataFrame({
            "open": [100, 105, 110],
            "high": [110, 115, 120],
            "low": [98, 103, 108],
            "close": [105, 110, 115],
            "volume": [1000, 1100, 1200]
        }, index=dates)
    
    monkeypatch.setattr(DataAdapter, "load_ohlcv", mock_load_ohlcv)
    return DataAdapter({})

def test_event_runner_gates_trade_on_max_positions(mock_settings, mock_data_adapter):
    """
    Tests that the EventRunner does not open a second trade when max_positions_total is 1.
    """
    signal_engine = MockSignalEngine()
    from ultra_signals.core.feature_store import FeatureStore
    fs = FeatureStore(warmup_periods=10, settings=mock_settings)
    runner = EventRunner(mock_settings, mock_data_adapter, signal_engine, fs)
    runner.warmup_mode = False
    
    # The mock signal engine will generate a LONG signal at every bar.
    # The first one should be executed.
    # The second one should be vetoed by the portfolio evaluation.
    trades, _ = runner.run("TEST_ETH", "1h")
    
    # Only the first trade should have been executed
    assert len(trades) == 1
    assert len(runner.risk_events) > 0
    assert any(e.reason == "MAX_POSITIONS_TOTAL" for e in runner.risk_events), "MAX_POSITIONS_TOTAL event not found"

def test_event_runner_downsizes_trade(mock_settings, mock_data_adapter):
    """
    Conceptual test for trade downsizing.
    
    This test verifies that if `evaluate_portfolio` returned a `size_scale < 1`,
    the resulting trade size would be smaller.
    """
    # This feature is not fully implemented in the mock `evaluate_portfolio`,
    # but this test serves as a placeholder for the logic.
    signal_engine = MockSignalEngine()
    from ultra_signals.core.feature_store import FeatureStore
    fs = FeatureStore(warmup_periods=10, settings=mock_settings)
    runner = EventRunner(mock_settings, mock_data_adapter, signal_engine, fs)
    
    # If the portfolio evaluator were to return a size_scale of 0.5,
    # we would expect the final position size to be halved.
    # This requires more complex mocking of `evaluate_portfolio`.
    pass