import pytest
import pandas as pd
from pathlib import Path
from ultra_signals.backtest.data_adapter import DataAdapter

@pytest.fixture(scope="module")
def sample_data_dir(tmpdir_factory):
    """Creates a temporary directory with sample data files."""
    tmp_path = tmpdir_factory.mktemp("data")
    
    # Create sample CSV
    csv_data = {
        "timestamp": pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 00:01:00"]),
        "open": [100, 101], "high": [102, 102], "low": [99, 100], "close": [101, 101.5], "volume": [10, 12]
    }
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(tmp_path.join("TEST_1m.csv"), index=False)

    # Create sample Parquet
    parquet_df = csv_df.copy()
    parquet_df = parquet_df.set_index('timestamp')
    parquet_df.to_parquet(tmp_path.join("TEST_1m.parquet"))

    return Path(str(tmp_path))

def test_load_ohlcv_from_csv(sample_data_dir):
    """Verify loading data from a CSV file."""
    config = {"data": {"provider": "csv", "base_path": str(sample_data_dir)}}
    adapter = DataAdapter(config)
    
    df = adapter.load_ohlcv("TEST", "1m", "2023-01-01", "2023-01-02")
    
    assert df is not None
    assert len(df) == 2
    assert "close" in df.columns
    assert df.index.name == "timestamp"

def test_load_ohlcv_from_parquet(sample_data_dir):
    """Verify loading data from a Parquet file."""
    config = {"data": {"provider": "parquet", "base_path": str(sample_data_dir)}}
    adapter = DataAdapter(config)
    
    df = adapter.load_ohlcv("TEST", "1m", "2023-01-01", "2023-01-02")
    
    assert df is not None
    assert len(df) == 2
    assert "open" in df.columns

def test_load_missing_file_returns_none(sample_data_dir):
    """Verify that a missing data file returns None."""
    config = {"data": {"provider": "csv", "base_path": str(sample_data_dir)}}
    adapter = DataAdapter(config)
    
    df = adapter.load_ohlcv("MISSING", "1m", "2023-01-01", "2023-01-02")
    assert df is None

def test_synthetic_trade_generation(sample_data_dir):
    """Verify the generation of synthetic trades from OHLCV data."""
    config = {"data": {"provider": "csv", "base_path": str(sample_data_dir)}}
    adapter = DataAdapter(config)
    
    trades = adapter.load_trades("TEST", "2023-01-01", "2023-01-02")
    
    assert trades is not None
    assert not trades.empty
    # Expect 4 trades per OHLCV row (2 rows * 4)
    assert len(trades) == 8
    assert "price" in trades.columns