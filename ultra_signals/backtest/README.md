# Backtest

This directory contains all components related to historical simulation and strategy evaluation.

- `event_runner.py`: An event-driven simulator that replays historical data through the same engine interface used by the real-time runner, ensuring consistency.
- `slippage.py`: Models for estimating and applying slippage to simulated trades based on order book or volatility data.
- `metrics.py`: Calculates and reports key performance indicators (KPIs) for a backtest, such as Sharpe ratio, drawdown, and profit factor.
- `walkforward.py`: A controller for running walk-forward optimizations, handling data purging and embargoing to prevent lookahead bias.