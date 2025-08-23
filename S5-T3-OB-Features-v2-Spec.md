# S5-T3: Order Book Features v2 Technical Specification

**Version:** 1.0
**Date:** 2025-08-22
**Status:** Draft

## 1. Overview

This document provides the technical specification for a set of advanced order book-based features designed to provide market confirmation signals. These features analyze liquidity and trade flow dynamics to generate insights into short-term price action.

## 2. Order Book Imbalance

### 2.1. Formula

Order Book Imbalance (OBI) is calculated as the ratio of the total volume on the bid side to the total volume on both the bid and ask sides within the top `N` levels of the order book.

The formula is:

```
TotalBidVolume = Σ(bid_price_i * bid_quantity_i) for i = 1 to N
TotalAskVolume = Σ(ask_price_i * ask_quantity_i) for i = 1 to N

ImbalanceRatio = TotalBidVolume / (TotalBidVolume + TotalAskVolume)
```

A value > 0.5 indicates heavier weight on the bid side (buying pressure), while a value < 0.5 indicates heavier weight on the ask side (selling pressure). A value of 0.5 represents a perfectly balanced book.

### 2.2. Configurable Parameters

*   `depth_levels_N`:
    *   **Description:** The number of order book levels (from the best bid/ask) to include in the calculation.
    *   **Type:** Integer
    *   **Default:** 10
    *   **Notes:** A smaller `N` focuses on immediate liquidity, while a larger `N` provides a broader view of the book depth.

*   `imbalance_ratio_threshold`:
    *   **Description:** The threshold to determine if the imbalance is significant. For example, a threshold of 0.6 would mean that the bid volume must constitute at least 60% of the total volume to be considered a strong buy-side imbalance.
    *   **Type:** Float
    *   **Default:** 0.6 (and by extension, 0.4 for the sell-side)

### 2.3. Example

Consider the following order book snapshot with `depth_levels_N = 5`:

**Bids:**
| Price | Quantity |
|-------|----------|
| 100   | 10       |
| 99    | 15       |
| 98    | 20       |
| 97    | 25       |
| 96    | 30       |

**Asks:**
| Price | Quantity |
|-------|----------|
| 101   | 5        |
| 102   | 10       |
| 103   | 15       |
| 104   | 20       |
| 105   | 25       |

**Calculation:**

*   `TotalBidVolume` = (100*10) + (99*15) + (98*20) + (97*25) + (96*30) = 1000 + 1485 + 1960 + 2425 + 2880 = **9750**
*   `TotalAskVolume` = (101*5) + (102*10) + (103*15) + (104*20) + (105*25) = 505 + 1020 + 1545 + 2080 + 2625 = **7775**
*   `ImbalanceRatio` = 9750 / (9750 + 7775) = 9750 / 17525 ≈ **0.556**

**Interpretation:**
The imbalance ratio of ~0.556 indicates a slight buying pressure, but it is below the default `imbalance_ratio_threshold` of 0.6, so it would not be flagged as a significant imbalance.

## 3. Book-Flip Detection

A "book-flip" is a significant and persistent change in the order book's liquidity dominance, transitioning from bid-heavy to ask-heavy, or vice-versa.

### 3.1. Detection Logic

A book-flip event is confirmed when the following conditions are met:

1.  **State Change:** The `ImbalanceRatio` crosses the 0.5 midpoint.
    *   **Ask-to-Bid Flip:** `ImbalanceRatio` moves from < 0.5 to > 0.5.
    *   **Bid-to-Ask Flip:** `ImbalanceRatio` moves from > 0.5 to < 0.5.
2.  **Minimum Delta:** The absolute change in the `ImbalanceRatio` between the old state and the new state must exceed the `minimum_delta` threshold.
    *   `abs(NewImbalanceRatio - OldImbalanceRatio) >= minimum_delta`
3.  **Persistence:** The new state (the `ImbalanceRatio` remaining above or below 0.5) must hold for a duration defined by the `persistence` parameter (e.g., a number of ticks or seconds).

The `OldImbalanceRatio` is recorded just before the state change.

### 3.2. Configurable Parameters

*   `minimum_delta`:
    *   **Description:** The minimum required change in the `ImbalanceRatio` to consider a flip significant. This prevents flagging minor oscillations around the midpoint.
    *   **Type:** Float
    *   **Default:** 0.15 (i.e., a 15% swing in the ratio)
*   `persistence`:
    *   **Description:** The duration (in seconds or number of data ticks) that the new imbalance state must be maintained for the flip to be considered valid.
    *   **Type:** Integer (e.g., seconds)
    *   **Default:** 5 (seconds)

### 3.3. Example Scenario

**Scenario:** Detect a flip from ask-dominant to bid-dominant.
**Parameters:** `minimum_delta = 0.15`, `persistence = 5s`.

*   **T=0s:** `ImbalanceRatio` is **0.35** (Ask-dominant). This is our `OldImbalanceRatio`.
*   **T=1s:** A large batch of buy limit orders arrives. The `ImbalanceRatio` jumps to **0.58**.
    *   **State Change:** Yes (0.35 -> 0.58, crosses 0.5).
    *   **Delta Check:** `abs(0.58 - 0.35) = 0.23`. Since `0.23 >= 0.15`, the delta condition is met.
    *   **Persistence Check:** The 5-second timer starts now.
*   **T=2s to T=6s:** The `ImbalanceRatio` fluctuates between 0.55 and 0.60, never dropping below 0.5.
*   **T=6s:** Since the `ImbalanceRatio` has remained > 0.5 for more than 5 seconds, a **valid Book-Flip event is confirmed**.

## 4. Simple Slippage Estimator

This feature estimates the price slippage for a hypothetical market order of a given size by calculating the Volume-Weighted Average Price (VWAP) required to fill that order.

### 4.1. Formula

To estimate the slippage for a **buy order** of `TradeSize`:

1.  Iterate through the ask book levels, starting from the best ask.
2.  For each level `i`, calculate the volume that can be filled: `FillVolume_i = min(RemainingTradeSize, AskQuantity_i)`.
3.  Calculate the cost at that level: `Cost_i = FillVolume_i * AskPrice_i`.
4.  Sum the `FillVolume_i` and `Cost_i` across all levels needed to fill the order.
5.  `VWAP_Fill = Σ(Cost_i) / Σ(FillVolume_i)`
6.  `Slippage = VWAP_Fill - BestAskPrice`

The logic is mirrored for a **sell order** using the bid book.

### 4.2. Example

**Scenario:** Estimate slippage for a market buy order of `TradeSize = 50`.

**Ask Book:**
| Price | Quantity | Cumulative Quantity |
|-------|----------|---------------------|
| 101   | 20       | 20                  |
| 102   | 25       | 45                  |
| 103   | 30       | 75                  |

**Calculation:**

1.  **Level 1 (Price 101):** We need 50, but only 20 are available.
    *   `FillVolume_1` = 20
    *   `Cost_1` = 20 * 101 = 2020
    *   `RemainingTradeSize` = 50 - 20 = 30
2.  **Level 2 (Price 102):** We need 30, and 25 are available.
    *   `FillVolume_2` = 25
    *   `Cost_2` = 25 * 102 = 2550
    *   `RemainingTradeSize` = 30 - 25 = 5
3.  **Level 3 (Price 103):** We need 5, and 30 are available.
    *   `FillVolume_3` = 5
    *   `Cost_3` = 5 * 103 = 515
    *   `RemainingTradeSize` = 5 - 5 = 0 (Order filled)

**VWAP and Slippage:**

*   `TotalCost` = 2020 + 2550 + 515 = **5085**
*   `TotalVolume` = 20 + 25 + 5 = **50**
*   `VWAP_Fill` = 5085 / 50 = **101.7**
*   `BestAskPrice` = 101
*   `Slippage` = 101.7 - 101 = **0.7**

## 5. Cumulative Volume Delta (CVD)

CVD tracks the net difference between buying and selling volume over time. It provides a raw look at whether buyers or sellers are more aggressive.

### 5.1. Formula

CVD is calculated based on executed trades (taker orders). Each trade is classified as a "taker buy" (aggressor hits the ask) or "taker sell" (aggressor hits the bid).

```
Delta_t = TakerBuyVolume_t - TakerSellVolume_t
CVD_t = CVD_{t-1} + Delta_t
```
Where:
*   `TakerBuyVolume_t` is the volume of aggressive buy orders in interval `t`.
*   `TakerSellVolume_t` is the volume of aggressive sell orders in interval `t`.
*   `CVD_t` is the cumulative sum.

### 5.2. CVD Slope

To measure the momentum of buying or selling pressure, we calculate the slope of the CVD over a recent lookback period. This can be done using a simple linear regression or a difference calculation.

**Simple Slope Formula:**

```
LookbackPeriod = K (e.g., number of recent data points or seconds)
CVD_Slope = (CVD_current - CVD_{current - K}) / K
```

A positive and rising slope indicates strengthening buying momentum. A negative and falling slope indicates strengthening selling momentum.
