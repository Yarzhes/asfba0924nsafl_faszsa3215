import pandas as pd
import pytest
from types import SimpleNamespace

from ultra_signals.backtest.json_metrics import build_run_metrics


def test_pnl_stratification_and_histograms():
    # Create synthetic trades with mtc_status tagging
    trades = pd.DataFrame([
        {"pnl": 10, "mtc_status": "CONFIRM"},
        {"pnl": -5, "mtc_status": "PARTIAL"},
        {"pnl": 7, "mtc_status": "CONFIRM"},
        {"pnl": -3, "mtc_status": "FAIL"},
        {"pnl": 4},  # untagged
    ])
    kpis = {"profit_factor":1.0, "sharpe":0.0, "max_drawdown":0.0, "win_rate_pct":0.0, "total_trades":5, "total_pnl": trades['pnl'].sum()}
    settings = {"backtest": {"execution": {"initial_capital": 10000}}}

    payload = build_run_metrics(kpis, trades, [10000,10005,10012], settings, "BTCUSDT", "5m")
    assert payload.get('pnl_mtc_confirm') == 17
    assert payload.get('pnl_mtc_partial') == -5
    assert payload.get('pnl_mtc_fail') == -3
    # counts
    assert payload.get('pnl_mtc_confirm_count') == 2
    assert payload.get('pnl_mtc_partial_count') == 1
    assert payload.get('pnl_mtc_fail_count') == 1


def test_observe_only_normalization():
    # Ensure observe-only logic in metrics: simulate trades where mtc_status exists but action was normalized
    trades = pd.DataFrame([
        {"pnl": 2, "mtc_status": "FAIL"},  # would be veto normally
    ])
    kpis = {"profit_factor":1.0, "sharpe":0.0, "max_drawdown":0.0, "win_rate_pct":0.0, "total_trades":1, "total_pnl": 2}
    settings = {"backtest": {"execution": {"initial_capital": 10000}}}
    payload = build_run_metrics(kpis, trades, [10000,10002], settings, "BTCUSDT", "5m")
    # Should still count pnl under fail bucket
    assert payload.get('pnl_mtc_fail') == 2
