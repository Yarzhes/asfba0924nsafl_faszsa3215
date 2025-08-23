# S4-T3: Liquidation Pulse Feature Specification

**Objective:** This document defines the specification for a "liquidation pulse" feature, which identifies and interprets significant liquidation events from a `forceOrder` data stream.

## 1. Rolling Window Definition

The feature will aggregate liquidation data over a configurable rolling window.

*   **Definition:** A rolling window is a fixed-duration time interval over which liquidation data is summed or counted. At each time step, the window slides forward, incorporating new data and dropping the oldest data.
*   **Default Window:** `5 minutes`
*   **Configuration:** The window size must be configurable to allow for tuning and optimization based on market conditions and asset volatility. Other suggested windows for analysis are `1 minute` and `15 minutes`.

## 2. Pulse Thresholds

A "pulse" is identified when aggregated liquidation metrics within the rolling window exceed predefined thresholds.

*   **Primary Metric:** Liquidation Volume (in USD).
*   **Secondary Metric:** Liquidation Count (number of individual liquidation events).

*   **Logic:** A pulse is triggered if **either** of the following conditions is met within the rolling window:
    1.  `Total Liquidation Volume` >= `Volume_Threshold`
    2.  `Total Liquidation Count` >= `Count_Threshold`

_The specific values for `Volume_Threshold` and `Count_Threshold` will be determined during the implementation and tuning phase but should be configurable._

## 3. Output Mapping

When a pulse is detected, the feature will generate an integer value and a "bias hint" to be stored in the `FeatureVector`. This provides a quantized representation of the pulse's intensity and its likely market impact.

| Pulse Intensity | Integer Value | Bias Hint      | Description                                                                                          |
| :-------------- | :------------ | :------------- | :--------------------------------------------------------------------------------------------------- |
| **High**        | `3`           | `mean_revert`  | A very strong liquidation cascade, often indicating a local top/bottom and a high probability of reversal. |
| **Medium**      | `2`           | `continuation` | A significant liquidation event that is likely to fuel the current short-term trend.               |
| **Low**         | `1`           | `continuation` | A minor liquidation event that confirms the current trend.                                           |
| **None**        | `0`           | `n/a`          | No significant liquidation activity detected.                                                        |

## 4. Labeled Examples

The following table provides hypothetical examples of how input data maps to the feature output.

| Timestamp Range (5-min window) | Total Liquidation Volume (USD) | Total Liquidation Count | Detected Pulse & Bias Hint | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `10:00 - 10:05` | `$5,000,000` | `350` | `3` (mean_revert) | Extreme volume and count suggest a capitulation event, ripe for reversal. |
| `10:05 - 10:10` | `$1,500,000` | `120` | `2` (continuation) | Strong liquidations, likely clearing out weaker hands and fueling trend continuation. |
| `10:10 - 10:15` | `$500,000` | `45` | `1` (continuation) | Moderate liquidations in line with an established trend. |
| `10:15 - 10:20` | `$50,000` | `5` | `0` (n/a) | Liquidation activity is below the minimum threshold for a pulse. |
| `10:20 - 10:25` | `$2,000,000` | `80` | `2` (continuation) | Volume threshold is met, indicating a significant but not climactic event. |
