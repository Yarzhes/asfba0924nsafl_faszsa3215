# S6-T4: Cost Model Specification

This document defines the cost models for transaction fees and funding rates to be used in the backtester. These models are designed to be configurable to allow for realistic simulation of trading costs.

## 1. Fee Model (Taker Fees)

This model calculates the cost of executing a trade.

*   **Fee Type:** Taker Fee
*   **Calculation:** The fee is calculated as a percentage of the notional value of the trade.
*   **Formula:** `Fee = TradeNotionalValue * TakerFeeRate`
*   **Configuration:**
    *   `TakerFeeRate`: A floating-point number representing the fee rate.
    *   **Default Value:** `0.0004` (equivalent to 0.04%)

## 2. Funding Rate Model

This model calculates the costs or payments accrued for holding a position overnight or over a funding interval.

*   **Logic:** The model processes a time series of historical funding rates for a given symbol. Funding is applied to the open position if a funding timestamp from the series falls within the duration of the current price bar.
*   **Calculation:** The funding cost/payment is calculated based on the position's notional value and the funding rate at that time.
*   **Formula:** `FundingCost = PositionNotionalValue * FundingRate`
*   **Timestamp-Logic:**
    *   Let `BarStartTime` and `BarEndTime` be the start and end timestamps of the current bar.
    *   Let `FundingTimestamp` be a timestamp from the funding rate trail.
    *   If `BarStartTime < FundingTimestamp <= BarEndTime`, then the funding rate corresponding to `FundingTimestamp` is applied to the position held at `BarEndTime`.

## 3. Assumptions

The following assumptions are made in the cost models:

1.  **All Orders are Taker Orders:** All simulated trades are assumed to be filled as taker orders, incurring the taker fee. Maker orders are not considered in this model.
2.  **Fees on Entry and Exit:** Taker fees are applied to both the entry and exit legs of a trade.
3.  **Funding on Open Positions:** Funding costs/payments are only applied to positions that are open at the time of a funding event.
4.  **Bar-Based Funding Application:** Funding is applied at the closing time of a bar if a funding timestamp falls within the bar's duration [start, end]. If multiple funding events occur within a single bar, all will be processed in chronological order.

## 4. Example Calculation

This section provides a step-by-step numerical example.

### Scenario

-   **Symbol:** BTC/USDT
-   **Taker Fee Rate:** 0.04%
-   **Initial Capital:** 10,000 USDT
-   **Funding Rates:**
    -   `2023-10-27 08:00:00 UTC`: +0.01%
    -   `2023-10-27 16:00:00 UTC`: +0.01%

### Step 1: Open Long Position

At `2023-10-27 04:00:00 UTC`, the strategy opens a long position.

-   **Trade Price:** 34,000 USDT
-   **Position Size:** 0.5 BTC
-   **Notional Value:** `0.5 BTC * 34,000 USDT/BTC = 17,000 USDT`

#### Fee Calculation (Entry)

-   **Fee Rate:** 0.04%
-   **Taker Fee:** `17,000 USDT * 0.0004 = 6.80 USDT`
-   **Cost Basis:** `17,000 USDT + 6.80 USDT = 17,006.80 USDT`

### Step 2: First Funding Event

A funding event occurs at `2023-10-27 08:00:00 UTC`. The position is open. The backtester is processing the 1-hour bar for `07:00` to `08:00`. The funding timestamp falls within this bar.

-   **Funding Rate:** +0.01% (Positive rate means longs pay shorts)
-   **Position Notional Value:** 17,000 USDT
-   **Funding Cost:** `17,000 USDT * 0.0001 = 1.70 USDT`
-   **Total Costs Accrued:** `6.80 USDT (fee) + 1.70 USDT (funding) = 8.50 USDT`

### Step 3: Second Funding Event (Position Still Open)

Another funding event occurs at `2023-10-27 16:00:00 UTC`. The position is still open.

-   **Funding Rate:** +0.01%
-   **Position Notional Value:** 17,000 USDT
-   **Funding Cost:** `17,000 USDT * 0.0001 = 1.70 USDT`
-   **Total Costs Accrued:** `8.50 USDT + 1.70 USDT = 10.20 USDT`

### Step 4: Close Long Position

At `2023-10-27 18:00:00 UTC`, the strategy closes the position.

-   **Trade Price:** 34,500 USDT
-   **Exit Notional Value:** `0.5 BTC * 34,500 USDT/BTC = 17,250 USDT`

#### Fee Calculation (Exit)

-   **Fee Rate:** 0.04%
-   **Taker Fee:** `17,250 USDT * 0.0004 = 6.90 USDT`

### Summary of Costs & PnL

-   **Gross PnL:** `17,250 USDT - 17,000 USDT = 250 USDT`
-   **Total Entry Fee:** 6.80 USDT
-   **Total Exit Fee:** 6.90 USDT
-   **Total Funding Costs:** `1.70 USDT + 1.70 USDT = 3.40 USDT`
-   **Total Costs:** `6.80 + 6.90 + 3.40 = 17.10 USDT`
-   **Net PnL:** `250 USDT - 17.10 USDT = 232.90 USDT`