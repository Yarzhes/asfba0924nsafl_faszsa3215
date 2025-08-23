# S5-T7: Risk Filters v2 Specification

## 1. Introduction

This document provides the technical specification for a new set of risk management filters designed to prevent the system from executing trades under high-risk market conditions. These filters leverage features derived from VWAP and the order book, as specified in `S5-T1-VWAP-Bands-Spec.md` and `S5-T3-OB-Features-v2-Spec.md`.

## 2. Filter Definitions

### 2.1. Breakout Confirmation Filter

*   **Objective:** To block breakout signals that are not confirmed by a contemporaneous "book-flip," indicating a shift in liquidity dominance that supports the breakout's direction.
*   **Logic:**
    1.  When a long breakout signal is generated, check for a recent "Ask-to-Bid" book-flip event.
    2.  When a short breakout signal is generated, check for a recent "Bid-to-Ask" book-flip event.
    3.  A book-flip is considered "recent" if it occurred within a configurable time window (`confirmation_window`) prior to the signal.
    4.  If the corresponding book-flip is not detected within the window, the trade is blocked.
*   **Dependencies:** Book-Flip Detection feature from `S5-T3-OB-Features-v2-Spec.md`.
*   **Configuration:**
    *   `confirmation_window`: The maximum time (in seconds) to look back for a confirming book-flip. Default: `10` seconds.

### 2.2. Mean-Reversion Filter

*   **Objective:** To ensure that mean-reversion trades are only taken when the price has extended to a statistically significant level away from the mean, as defined by VWAP standard deviation bands.
*   **Logic:**
    1.  When a long mean-reversion signal is generated (expecting price to rise), the filter checks if the current price (`Close` or `Low`) has pierced below a configurable lower VWAP band.
    2.  When a short mean-reversion signal is generated (expecting price to fall), the filter checks if the current price (`Close` or `High`) has pierced above a configurable upper VWAP band.
    3.  If the condition is not met, the trade is blocked.
*   **Dependencies:** VWAP & Bands feature from `S5-T1-VWAP-Bands-Spec.md`.
*   **Configuration:**
    *   `vwap_type`: The type of VWAP to use (`Rolling`, `Session`, or `Anchored`).
    *   `band_multiplier`: The standard deviation band multiplier that must be pierced for the trade to be allowed (e.g., `2.0` for the 2nd standard deviation band).

### 2.3. CVD Alignment Filter (Optional)

*   **Objective:** To provide an optional layer of confirmation by ensuring that the short-term order flow momentum, measured by the CVD slope, aligns with the intended trade direction.
*   **Logic:**
    1.  This filter is optional and can be enabled or disabled via configuration.
    2.  If enabled, when a **long** signal is generated, the filter checks if the `CVD_Slope` over a recent lookback period is positive.
    3.  If enabled, when a **short** signal is generated, the filter checks if the `CVD_Slope` over a recent lookback period is negative.
    4.  If the slope is not aligned with the trade direction, the trade is blocked.
*   **Dependencies:** CVD Slope feature from `S5-T3-OB-Features-v2-Spec.md`.
*   **Configuration:**
    *   `enabled`: `true` or `false`. Default: `false`.
    *   `lookback_period_K`: The lookback period for calculating the CVD slope.

### 2.4. Slippage Cap Filter

*   **Objective:** To protect against excessive transaction costs by blocking trades where the estimated slippage exceeds a predefined maximum threshold.
*   **Logic:**
    1.  Before placing a trade, the system calculates the estimated slippage using the Simple Slippage Estimator for the intended trade size.
    2.  The estimated slippage is compared against a configurable `max_slippage_bps` (basis points) or `max_slippage_usd` value.
    3.  If the estimated slippage is greater than the configured maximum, the trade is blocked.
*   **Dependencies:** Simple Slippage Estimator from `S5-T3-OB-Features-v2-Spec.md`.
*   **Configuration:**
    *   `max_slippage_bps`: The maximum allowable slippage in basis points (e.g., `5` bps).
    *   `trade_size_source`: Where to get the trade size for estimation (e.g., from the sizing module).

---

## 3. Test Cases

### 3.1. Test Cases for Breakout Confirmation Filter

| Test Case ID      | Market Conditions                                                                                                       | Signal              | Expected Behavior | Reason                                                                                 |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------- | ----------------- | -------------------------------------------------------------------------------------- |
| BC-PASS-LONG      | Price breaks above resistance. An "Ask-to-Bid" book-flip was confirmed 3 seconds ago.                                     | Long Breakout       | **Allowed**       | The breakout is confirmed by a recent and valid shift in order book liquidity.         |
| BC-FAIL-LONG-1    | Price breaks above resistance. No book-flip has occurred in the last 10 seconds.                                        | Long Breakout       | **Blocked**       | The breakout is not supported by a corresponding shift in the order book.              |
| BC-FAIL-LONG-2    | Price breaks above resistance. A "Bid-to-Ask" book-flip (opposite direction) was confirmed 5 seconds ago.                 | Long Breakout       | **Blocked**       | The order book liquidity is shifting against the direction of the breakout.            |
| BC-PASS-SHORT     | Price breaks below support. A "Bid-to-Ask" book-flip was confirmed 2 seconds ago.                                       | Short Breakout      | **Allowed**       | The breakout is confirmed by a recent and valid shift in order book liquidity.         |
| BC-FAIL-SHORT-1   | Price breaks below support. The last book-flip was an "Ask-to-Bid" event 15 seconds ago (outside the confirmation window). | Short Breakout      | **Blocked**       | No confirming book-flip occurred within the required `confirmation_window`.            |

### 3.2. Test Cases for Mean-Reversion Filter

(Using `band_multiplier = 2.0`)

| Test Case ID      | Market Conditions                                                                       | Signal                    | Expected Behavior | Reason                                                                             |
| ----------------- | --------------------------------------------------------------------------------------- | ------------------------- | ----------------- | ---------------------------------------------------------------------------------- |
| MR-PASS-LONG      | Price drops and the Low of the current bar is below the 2nd lower VWAP band.              | Long Mean-Reversion       | **Allowed**       | The price has reached a statistically significant deviation below the VWAP mean.   |
| MR-FAIL-LONG      | Price is trading between the 1st and 2nd lower VWAP bands, but has not pierced the 2nd. | Long Mean-Reversion       | **Blocked**       | The price has not extended far enough from the mean to justify a reversion trade.  |
| MR-PASS-SHORT     | Price rallies and the High of the current bar is above the 2nd upper VWAP band.           | Short Mean-Reversion      | **Allowed**       | The price has reached a statistically significant deviation above the VWAP mean.     |
| MR-FAIL-SHORT     | Price is trading just above the 1st upper VWAP band.                                    | Short Mean-Reversion      | **Blocked**       | The price has not extended far enough from the mean to justify a reversion trade.  |

### 3.3. Test Cases for CVD Alignment Filter

(Assuming filter is enabled)

| Test Case ID      | Market Conditions                               | Signal              | Expected Behavior | Reason                                                                       |
| ----------------- | ----------------------------------------------- | ------------------- | ----------------- | ---------------------------------------------------------------------------- |
| CVD-PASS-LONG     | CVD slope over the lookback period is `+500`.     | Long Signal         | **Allowed**       | The positive CVD slope indicates buying pressure, aligning with a long trade.    |
| CVD-FAIL-LONG     | CVD slope over the lookback period is `-250`.     | Long Signal         | **Blocked**       | The negative CVD slope indicates selling pressure, contradicting a long trade.  |
| CVD-PASS-SHORT    | CVD slope over the lookback period is `-700`.     | Short Signal        | **Allowed**       | The negative CVD slope indicates selling pressure, aligning with a short trade.  |
| CVD-FAIL-SHORT    | CVD slope over the lookback period is `+100`.     | Short Signal        | **Blocked**       | The positive CVD slope indicates buying pressure, contradicting a short trade. |
| CVD-PASS-DISABLED | CVD slope is `-1000`, filter is disabled.       | Long Signal         | **Allowed**       | The filter is optional and currently disabled, so it does not block the trade. |

### 3.4. Test Cases for Slippage Cap Filter

(Using `max_slippage_bps = 5`)

| Test Case ID      | Market Conditions                          | Estimated Slippage (bps) | Expected Behavior | Reason                                                                   |
| ----------------- | ------------------------------------------ | ------------------------ | ----------------- | ------------------------------------------------------------------------ |
| SLIP-PASS-LOW     | Liquid market, thin spread for trade size. | 2 bps                    | **Allowed**       | The estimated slippage is within the acceptable `max_slippage_bps` limit.  |
| SLIP-PASS-EDGE    | Moderate liquidity.                          | 5 bps                    | **Allowed**       | The estimated slippage is equal to the maximum allowed limit.              |
| SLIP-FAIL-HIGH    | Illiquid market, wide spread for trade size. | 12 bps                   | **Blocked**       | The estimated slippage exceeds the `max_slippage_bps` threshold.           |