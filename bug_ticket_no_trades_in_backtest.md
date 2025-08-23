Title: [Backtest] Backtest not generating meaningful trades or PnL
Steps:
1. Run backtest with `python -m ultra_signals.apps.backtest_cli run --config settings.yaml --output-dir reports/s6_run1`
2. Run second backtest with `python -m ultra_signals.apps.backtest_cli run --config settings.yaml --output-dir reports/s6_run2`
3. Observe `summary.txt` in both output directories.
Expected: Backtests should generate multiple trades and a non-zero PnL, with KPIs reflecting actual trading performance.
Actual: Both backtests show `total_pnl: 0` and `total_trades: 1`, with `win_rate_pct: 0.0000`, `profit_factor: inf`, `sharpe_ratio: nan`, `average_win: 0`, and `average_loss: 0`.
Logs/Screens:
- `reports/s6_run1/summary.txt`
- `reports/s6_run2/summary.txt`
Scope: backtest
Suspected cause: Issue in `EventRunner` or `RealSignalEngine` preventing trade generation or proper PnL calculation. Could be related to signal generation, risk filters, or sizing.
Severity: blocker
Fix idea: Investigate `ultra_signals/backtest/event_runner.py` and `ultra_signals/engine/real_engine.py` to understand why trades are not being generated or processed correctly. Check signal thresholds, risk management rules, and sizing logic.
Owner: Debug