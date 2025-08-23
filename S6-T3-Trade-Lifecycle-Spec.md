# S6-T3: Trade Lifecycle Simulation Rules

This document specifies the rules governing trade execution, management, and closure within the backtesting simulation. The definitions herein are designed to be unambiguous, prevent look-ahead bias, and ensure a realistic simulation of trading strategy mechanics.

## 1. Entry Logic

This section defines how trades are initiated based on signals.

### 1.1. Limit Order: Zone Touch

A limit order is considered filled if the price bar touches or crosses the specified entry price.

-   **For a Long Position:** The trade is filled if `Bar.Low <= EntryPrice`. The execution price is the `EntryPrice`.
-   **For a Short Position:** The trade is filled if `Bar.High >= EntryPrice`. The execution price is the `EntryPrice`.
-   **Condition:** This logic assumes the signal was generated on the *previous* bar. The check occurs on the current bar.

### 1.2. Market Order: Execute at Next Open

A market order is executed at the opening price of the bar immediately following the signal bar.

-   **Execution Price:** `NextBar.Open`.
-   **Condition:** The signal is generated based on the data of `Bar[T-1]`. The trade is executed at the open of `Bar[T]`.

## 2. Trade Management Logic

This section outlines the rules for managing open positions.

### 2.1. Partial Take-Profit (TP1)

When the `TP1` level is hit, a predefined portion of the position is closed.

-   **Rule:** If `TP1` is touched or crossed, close X% of the original position size (e.g., 50%).
-   **Execution Price:** The `TP1` price level.
-   **Remaining Position:** The stop-loss for the remaining portion of the trade is adjusted as per the Break-Even rule (see 2.2).

### 2.2. Stop-Loss to Break-Even (BE)

After `TP1` is hit and a partial profit is taken, the stop-loss for the remainder of the position is moved to the original entry price.

-   **Trigger:** Successful execution of the `TP1` partial close.
-   **Action:** The `StopLoss` for the remaining position is set to the original `EntryPrice`.
-   **Purpose:** This secures the trade, ensuring that the worst-case outcome for the remaining position is a scratch (zero loss).

### 2.3. Trailing Stop-Loss (TSL)

A trailing stop-loss dynamically adjusts the stop-loss level to lock in profits as the price moves favorably.

-   **Mechanism:** The stop-loss is trailed by a fixed percentage or a multiple of the Average True Range (ATR) from the current price.
-   **Example (ATR-based):** `NewSL = CurrentPrice - (N * ATR)`. The `NewSL` is only adjusted if it is higher (for longs) or lower (for shorts) than the previous `SL`.
-   **Activation:** The TSL can be configured to activate immediately upon trade entry or after `TP1` is hit. This must be specified by the strategy.

### 2.4. Cooldown Period

After a trade is closed (either by `SL`, `TP`, or `TSL`), a cooldown period prevents re-entry on the same symbol for a specified number of bars.

-   **Rule:** No new trades can be initiated on `Symbol X` for `N` bars following the closure of a previous trade on `Symbol X`.
-   **Purpose:** Prevents immediate re-entry on volatile "whipsaw" price action and allows the market state to reset before considering a new signal.

## 3. Intra-Bar Execution Order

This section defines the priority of execution when multiple events occur within the same price bar, which is critical for resolving conflicts between Stop-Loss and Take-Profit levels.

### 3.1. Execution Priority

To ensure consistent and conservative backtest results, the following order of operations is strictly enforced for events occurring within a single price bar:

1.  **Stop-Loss (SL) Execution:** The Stop-Loss is always checked and executed first.
2.  **Take-Profit (TP) Execution:** The Take-Profit is checked and executed only if the Stop-Loss was not triggered in the same bar.

### 3.2. Rationale

This "Stop-Loss First" rule is a cornerstone of prudent risk management in simulation. The rationale is as follows:

-   **Conservatism:** A bar that touches both the SL and TP levels is inherently volatile. In a live trading scenario, there is no guarantee which level was hit first without tick-level data. Assuming the SL was hit first represents the worst-case, most conservative outcome, which prevents the overestimation of strategy performance.
-   **Prevents Look-Ahead Bias:** By not assuming the more favorable outcome (TP), we avoid introducing look-ahead bias into the simulation. The OHLC data of a bar summarizes its activity, but not the sequence of that activity. Prioritizing the SL is a robust guard against making unsafe assumptions about intra-bar price travel.
-   **Industry Standard:** This approach is a widely accepted standard in backtesting platforms to ensure that simulated results are more likely to align with real-world performance.

## 4. Truth Table of Edge Cases

This table provides a definitive outcome for various intra-bar scenarios for an **active Long position**. The logic is mirrored for short positions.

| Scenario ID | Bar Condition (`Low` & `High`) | Entry Price | SL Price | TP Price | Outcome | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **L-1** | `Low > SL` and `High < TP` | 100 | 95 | 105 | **No Action** | The bar's range is entirely within the SL/TP levels. |
| **L-2** | `Low <= SL` and `High < TP` | 100 | 95 | 105 | **Stop-Loss Hit** | The SL was touched or breached. The trade is closed at the SL price. |
| **L-3** | `Low > SL` and `High >= TP` | 100 | 95 | 105 | **Take-Profit Hit** | The TP was touched or breached. The trade is closed at the TP price. |
| **L-4** | `Low <= SL` and `High >= TP` | 100 | 95 | 105 | **Stop-Loss Hit** | **[Key Scenario]** Both levels were breached. Per section 3.1, the SL is prioritized. |
| **L-5** | `High >= Entry` and `Low <= SL` | - | 95 | - | **Entry -> Stop-Loss** | A limit order is filled and then stopped out in the same bar. SL is prioritized. |
| **L-6** | `High >= TP` and `Low <= Entry` | - | - | 105 | **Entry -> Take-Profit** | A limit order is filled and then hits the take-profit in the same bar. |
