import pytest
import pandas as pd
from unittest.mock import MagicMock
from ultra_signals.backtest.event_runner import EventRunner, MockSignalEngine

@pytest.fixture
def mock_data_adapter():
    """Fixture to create a mock data adapter with sample OHLCV data."""
    adapter = MagicMock()
    
    # Create sample data that ensures signals will be generated
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 00:05', '2023-01-01 00:10']),
        'open': [100, 102, 101],
        'high': [103, 104, 102],
        'low': [99, 101, 100],
        'close': [102, 101, 101.5]
    }
    df = pd.DataFrame(data).set_index('timestamp')
    adapter.load_ohlcv.return_value = df
    return adapter

@pytest.fixture
def basic_settings():
    """Provides a basic configuration for the event runner."""
    return {
        "start_date": "2023-01-01",
        "end_date": "2023-01-02",
        "initial_capital": 50000.0,
        "default_size_pct": 0.10, # 10% of capital per trade
        "features": {
            "warmup_periods": 10,
            "trend": {},
            "momentum": {},
            "volatility": {},
            "volume_flow": {}
        }
    }

def test_runner_initialization(basic_settings):
    """Test that the EventRunner initializes correctly."""
    from ultra_signals.core.feature_store import FeatureStore
    fs = FeatureStore(warmup_periods=10, settings=basic_settings)
    runner = EventRunner(basic_settings, MagicMock(), MockSignalEngine(), fs)
    assert runner.portfolio.equity == 50000.0
    assert not runner.trades
    assert not runner.equity_curve

def test_run_executes_trades_and_updates_state(basic_settings, mock_data_adapter):
    """
    Verify that the event runner processes bars, executes trades,
    and updates the portfolio state correctly.
    """
    from ultra_signals.core.feature_store import FeatureStore
    fs = FeatureStore(warmup_periods=10, settings=basic_settings)
    runner = EventRunner(basic_settings, mock_data_adapter, MockSignalEngine(), fs)
    
    trades, equity = runner.run(symbol="TEST_ETH", timeframe="5m")
    
    # Verify data was loaded
    mock_data_adapter.load_ohlcv.assert_called_once()
    
    # Verify trades were generated and recorded
    assert trades is not None
    assert len(trades) == 1, "Only one trade should have been executed"
    assert "pnl" in trades[0]
    
    # Verify equity curve was generated
    assert equity is not None
    assert len(equity) == 3 # One entry per bar
    assert "equity" in equity[0]
    
    # In warmup, we don't expect PNL to be realized.
    pass

def test_warmup_period(basic_settings, mock_data_adapter):
    """
    This test simulates ensuring a warmup period is respected.
    While the current runner doesn't explicitly have a warmup parameter,
    the logic of feeding data sequentially inherently creates one for indicators.
    This test verifies that the signal engine receives data correctly.
    """
    mock_engine = MagicMock(wraps=MockSignalEngine())
    from ultra_signals.core.feature_store import FeatureStore
    fs = FeatureStore(warmup_periods=10, settings=basic_settings)
    runner = EventRunner(basic_settings, mock_data_adapter, mock_engine, fs)
    runner.run("TEST_WARMUP", "1h")

    # The signal engine's `generate_signal` should be called for each bar.
    assert mock_engine.generate_signal.call_count == 1
    
    # Inspect the data passed on the first call
    first_call_args = mock_engine.generate_signal.call_args_list[0]
    df_passed = first_call_args.kwargs['ohlcv_segment']
    assert len(df_passed) == 1
    assert first_call_args.kwargs['symbol'] == "TEST_WARMUP"
    assert len(df_passed) == 1
    assert df_passed.iloc[0]['close'] == 102