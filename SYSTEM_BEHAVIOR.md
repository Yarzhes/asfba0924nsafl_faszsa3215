# System Behavior and Performance

This document specifies the core execution logic, determinism guarantees, and performance targets for the backtesting system.

## Execution Rules

The following rules ensure that the simulation is deterministic and handles intrabar events consistently.

1.  **Intrabar SL/TP Ordering**:
    - When both a Stop Loss (SL) and Take Profit (TP) level are breached within the same candle (i.e., `high >= tp` and `low <= sl`), the executed price is determined by the `sl_tp_order` setting:
        - `"SL_first"`: The Stop Loss is always triggered first.
        - `"TP_first"`: The Take Profit is always triggered first.
        - `"intrabar_both"` (Default): The outcome is determined by which level was closer to the entry price. The engine checks `min(abs(entry - high), abs(entry - low))` to decide which was likely hit first.

2.  **Partial Take Profit**:
    - If `partial_tp: true`, the position is managed as follows:
        - At TP1, 50% of the position size is closed.
        - The Stop Loss for the remaining 50% is moved to the break-even price.
        - The remainder of the position is trailed according to the `trail_mode` setting until the trailing stop is hit or the backtest ends.

3.  **Costs Application**:
    - **Fees**: `taker_fee_bps` is applied to the notional value of every fill (entry, partial TP, and final exit).
    - **Slippage**: The slippage model is invoked on every fill to adjust the execution price.
    - **Funding**: For perpetual contracts, funding is calculated and applied every 8 hours, based on the open position at the funding timestamp. The `funding_model` approximates this cost.

## Performance & Reliability

-   **Performance Budget**: A backtest of 10 symbols on the 5-minute timeframe over a 1-year period should complete in approximately 10-20 minutes on a standard developer machine. This requires efficient, vectorized operations and chunked data I/O to avoid memory bottlenecks.
-   **Determinism**: The entire backtesting and walk-forward pipeline must be fully deterministic. For a given configuration file and dataset, the output artifacts (reports, trades, equity curves) must be identical across multiple runs. All sources of randomness, including model seeds and simulation sampling (if any), must be controlled by fixed seeds.