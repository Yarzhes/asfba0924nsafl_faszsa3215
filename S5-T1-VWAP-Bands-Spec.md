# S5-T1: VWAP & Bands Technical Specification

## 1. Introduction

This document provides the technical specification for the Volume-Weighted Average Price (VWAP) and its associated standard deviation bands. It covers three distinct VWAP implementations: Rolling, Session, and Anchored.

## 2. General Concepts

### 2.1. Typical Price

The typical price is a fundamental component of the VWAP calculation. For any given bar `i`, it is calculated as:

`Typical Price (TP_i) = (High_i + Low_i + Close_i) / 3`

### 2.2. Standard Deviation Bands

Standard deviation bands are used to measure volatility around the VWAP. For each VWAP type, `k` standard deviation bands will be calculated above and below the VWAP line.

- `Upper Band (k) = VWAP + (Multiplier_k * Standard_Deviation)`
- `Lower Band (k) = VWAP - (Multiplier_k * Standard_Deviation)`

Common multipliers are 1, 2, and 3, but these will be configurable.

## 3. Rolling VWAP

### 3.1. Formula

The Rolling VWAP over a lookback period of `N` bars is calculated as follows:

`Rolling VWAP = Σ(TP_i * V_i) / Σ(V_i)` for `i` over the last `N` bars.

Where:
- `TP_i` = Typical Price of bar `i`
- `V_i` = Volume of bar `i`

### 3.2. Standard Deviation

The standard deviation of the Rolling VWAP measures the dispersion of price from the VWAP over the same `N`-bar lookback period.

1.  **Calculate the squared difference for each bar:**
    `Squared Difference_i = (Close_i - Rolling VWAP)^2`

2.  **Calculate the Volume-Weighted Variance:**
    `Variance = Σ(Squared Difference_i * V_i) / Σ(V_i)` for `i` over the last `N` bars.

3.  **Calculate the Standard Deviation:**
    `Standard Deviation = sqrt(Variance)`

### 3.3. Configuration

-   `lookback_period`: The `N` number of bars for the rolling calculation.
-   `band_multipliers`: A list of numbers for the standard deviation bands (e.g., `[1, 2, 3]`).

### 3.4. Warm-up Period

The Rolling VWAP requires `N` bars of data to be available before the first value can be calculated. The indicator should not produce any output until this condition is met.

## 4. Session VWAP

### 4.1. Formula & Reset Logic

The Session VWAP calculates the volume-weighted average price from the start of a defined trading session. The calculation is cumulative and resets at the beginning of each new session.

`Session VWAP = Σ(TP_i * V_i) / Σ(V_i)` for all bars `i` since the start of the current session.

- **Reset Logic:** The cumulative sums `Σ(TP_i * V_i)` and `Σ(V_i)` are reset to zero at the first bar of each new session.

### 4.2. Standard Deviation

The standard deviation for the Session VWAP is calculated similarly to the Rolling VWAP, but the calculation is cumulative over the session.

1.  **Calculate the squared difference for each bar in the session:**
    `Squared Difference_i = (Close_i - Session VWAP)^2`

2.  **Calculate the cumulative Volume-Weighted Variance for the session:**
    `Variance = Σ(Squared Difference_i * V_i) / Σ(V_i)` for all bars `i` since the session start.

3.  **Calculate the Standard Deviation:**
    `Standard Deviation = sqrt(Variance)`

### 4.3. Configuration

-   `sessions`: A list of session definitions, each with a `name`, `start_time`, and `end_time` (UTC).
-   `band_multipliers`: A list of numbers for the standard deviation bands (e.g., `[1, 2, 3]`).

### 4.4. Session Definitions

Sessions are defined by a start and end time in UTC. The VWAP calculation will be active for any bars falling within these time ranges.

**Example Configuration:**

-   **Asia Session:** 00:00 - 08:00 UTC
-   **London Session:** 07:00 - 16:00 UTC
-   **New York Session:** 13:00 - 21:00 UTC

**Note:** Sessions can overlap. The system must correctly handle bars that may fall into more than one active session. Each session will have its own independent VWAP calculation.

### 4.5. Warm-up Period

The Session VWAP does not have a fixed warm-up period. The calculation begins with the first bar of the session. The VWAP value will become more statistically significant as more bars accumulate within the session.

## 5. Anchored VWAP

### 5.1. Formula & Anchor Logic

The Anchored VWAP calculates the VWAP from a specific anchor point (a bar or a time). The calculation is cumulative from the anchor point until a new anchor point is identified or the conditions for the anchor are no longer met.

`Anchored VWAP = Σ(TP_i * V_i) / Σ(V_i)` for all bars `i` from the anchor bar to the current bar.

- **Anchor Logic:** The calculation begins at a specified anchor bar. All subsequent bars are included in the cumulative calculation until a new anchor is set.

### 5.2. Standard Deviation

The standard deviation for the Anchored VWAP is calculated cumulatively from the anchor point.

1.  **Calculate the squared difference for each bar since the anchor:**
    `Squared Difference_i = (Close_i - Anchored VWAP)^2`

2.  **Calculate the cumulative Volume-Weighted Variance:**
    `Variance = Σ(Squared Difference_i * V_i) / Σ(V_i)` for all bars `i` from the anchor.

3.  **Calculate the Standard Deviation:**
    `Standard Deviation = sqrt(Variance)`

### 5.3. Configuration

-   `anchor_type`: The type of anchor to use (e.g., `DAILY_OPEN`, `SWING_HIGH`, `SWING_LOW`).
-   `swing_lookback`: The number of bars to look back for identifying swing points (only for swing anchor types).
-   `band_multipliers`: A list of numbers for the standard deviation bands.

### 5.4. Anchor Types

#### 5.4.1. Daily Open

-   **Anchor Point:** The first bar of the trading day (e.g., at 00:00 UTC).
-   **Logic:** The VWAP calculation starts at the open of the daily bar and continues cumulatively until the end of the day. It resets at the start of the next day.

#### 5.4.2. Recent Swing High/Low

-   **Anchor Point:** A recent significant swing high or swing low point.
-   **Logic:**
    -   A **Swing High** is identified as the highest high within a `swing_lookback` period, provided it is also higher than the bars immediately preceding and succeeding it (e.g., `High[i-1] < High[i] > High[i+1]`).
    -   A **Swing Low** is identified as the lowest low within a `swing_lookback` period, provided it is also lower than the bars immediately preceding and succeeding it (e.g., `Low[i-1] > Low[i] < Low[i+1]`).
-   **Refresh Logic:** The system will continuously monitor for new swing points. When a new valid swing point is confirmed, the anchor is moved, and the VWAP calculation resets.

### 5.5. Warm-up Period

The Anchored VWAP begins calculation from the anchor bar. No warm-up period is required past the anchor point itself. However, for `SWING_HIGH`/`SWING_LOW` anchors, a `swing_lookback` number of bars is required to identify the first anchor.

## 6. Edge Cases

### 6.1. Price Gaps

-   **Definition:** A significant price difference between the close of one bar and the open of the next.
-   **Handling:** The calculations for TP, VWAP, and Standard Deviation are based on the HLC (High, Low, Close) and Volume of each bar. Price gaps are inherently handled by this bar-based calculation, as the TP of the new bar will correctly reflect the gapped price. No special logic is required.

### 6.2. Anchor Refresh Logic

-   **For Swing High/Low:** The system must check for a new swing point on the close of every bar. To prevent unstable anchors, a new swing high/low must be confirmed before the anchor is moved. A simple confirmation is to wait for one or two bars to close after the potential swing point. The exact number of confirmation bars should be configurable.

### 6.3. Insufficient Data

-   **At Startup:** The system must wait for the required warm-up data before producing any VWAP values (e.g., `N` bars for Rolling VWAP).
-   **For Session/Anchored:** If the data source provides no bars for a long period within a session or after an anchor, the VWAP will remain flat. This is expected behavior.