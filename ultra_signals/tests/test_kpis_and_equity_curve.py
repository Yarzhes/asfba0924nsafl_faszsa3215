import pytest
import pandas as pd
import numpy as np
from ultra_signals.backtest.metrics import compute_kpis, generate_equity_curve

@pytest.fixture
def sample_trades():
    """Provides a sample DataFrame of trades for testing."""
    trades_data = {
        'exit_time': pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04']),
        'pnl': [100.0, -50.0, 200.0]
    }
    return pd.DataFrame(trades_data)

def test_compute_kpis_happy_path(sample_trades):
    """Test KPI computation with a typical set of trades."""
    kpis = compute_kpis(sample_trades)
    
    assert kpis['total_pnl'] == 250
    assert kpis['total_trades'] == 3
    assert pytest.approx(kpis['win_rate_pct']) == (2/3) * 100
    assert pytest.approx(kpis['profit_factor']) == 300 / 50 # 6.0
    assert kpis['average_win'] == 150 # (100 + 200) / 2
    assert kpis['average_loss'] == -50

def test_compute_kpis_no_trades():
    """Test that KPI computation handles the case with no trades."""
    kpis = compute_kpis(pd.DataFrame())
    assert "error" in kpis

def test_generate_equity_curve(sample_trades):
    """Test the generation of an equity curve."""
    initial_capital = 1000
    equity_curve = generate_equity_curve(sample_trades, initial_capital)
    
    assert isinstance(equity_curve, pd.Series)
    assert len(equity_curve) == 3
    
    # Expected equity: 1000 + 100 = 1100, 1100 - 50 = 1050, 1050 + 200 = 1250
    expected_equity = pd.Series([1100.0, 1050.0, 1250.0], index=sample_trades['exit_time'])
    pd.testing.assert_series_equal(equity_curve, expected_equity, check_names=False)