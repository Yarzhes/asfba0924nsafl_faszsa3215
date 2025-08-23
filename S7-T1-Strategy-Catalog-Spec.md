# S7-T1: Strategy Catalog & Outputs Specification

This document defines the specifications for four distinct trading strategies that will be part of the Ultra-Signals ensemble. For each strategy, we detail its required inputs, triggering conditions, and a standardized output object.

## Standardized Signal Output

Each strategy, upon identifying a valid trading opportunity, will generate a sub-signal output object with the following structure:

```json
{
  "direction": "long | short",
  "conf_calibrated": 0.0,
  "reasons": ["reason_1", "reason_2"],
  "strategy_id": "strategy_name"
}
```

-   **`direction`**: The direction of the anticipated price movement (`long` or `short`).
-   **`conf_calibrated`**: A calibrated confidence score between 0.0 and 1.0.
-   **`reasons`**: A list of strings describing the specific conditions that triggered the signal.
-   **`strategy_id`**: The unique identifier for the strategy that generated the signal.

---

## Strategy Definitions

The following table summarizes the inputs and triggering conditions for each of the four strategies.

| Strategy ID         | Description                                                      | Required Inputs                                                                                                | Triggering Conditions                                                                                                                                                             |
| ------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `trend_pullback`    | An EMA/VWAP pullback in a trending market.                       | <ul><li>Price (Candles)</li><li>Fast EMA</li><li>Slow EMA</li><li>VWAP</li></ul>                               | <ul><li>**Trend ID:** Fast EMA is above Slow EMA (for long) or below (for short).</li><li>**Pullback:** Price touches or pulls back to the Fast EMA/VWAP.</li></ul>                   |
| `breakout_book`     | A price level break with an order book flip and CVD confirmation.  | <ul><li>Price (Candles)</li><li>Significant Price Levels (e.g., daily hi/lo)</li><li>Order Book Data</li><li>CVD</li></ul> | <ul><li>**Break:** Price breaks a significant level.</li><li>**OB Flip:** Order book imbalance flips to support the breakout direction.</li><li>**CVD Conf:** CVD confirms the move.</li></ul> |
| `mean_revert_vwap`  | A price piercing a VWAP band followed by a reversion signal.     | <ul><li>Price (Candles)</li><li>VWAP</li><li>VWAP Bands (e.g., 2 std dev)</li></ul>                           | <ul><li>**Pierce:** Price pierces an outer VWAP band.</li><li>**Reversion:** Price closes back inside the band, signaling reversion to the mean (VWAP).</li></ul>                       |
| `sweep_reversal`    | A liquidation pulse creates a wick, followed by a reversal.      | <ul><li>Price (Candles)</li><li>Liquidation Data Feed</li><li>Volume</li></ul>                                  | <ul><li>**Sweep:** A high-volume liquidation cascade creates a significant price wick.</li><li>**Reversal:** Price begins to reverse direction after the sweep.</li></ul>                |

---

## Sub-Signal Output Examples

### 1. `trend_pullback`

**Scenario:** The market is in an uptrend, and the price pulls back to the 20-period EMA.

```json
{
  "direction": "long",
  "conf_calibrated": 0.75,
  "reasons": ["ema_fast > ema_slow", "price_touch_ema_fast"],
  "strategy_id": "trend_pullback"
}
```

### 2. `breakout_book`

**Scenario:** Price breaks above yesterday's high, the order book shows strong buy-side pressure, and CVD is rising.

```json
{
  "direction": "long",
  "conf_calibrated": 0.82,
  "reasons": ["break_daily_high", "ob_imbalance_flip_bullish", "cvd_conf_bullish"],
  "strategy_id": "breakout_book"
}
```

### 3. `mean_revert_vwap`

**Scenario:** Price moves sharply down, piercing the lower VWAP band, and then a bullish candle closes back inside the band.

```json
{
  "direction": "long",
  "conf_calibrated": 0.68,
  "reasons": ["price_pierce_vwap_lower_band", "reversion_candle_close_inside"],
  "strategy_id": "mean_revert_vwap"
}
```

### 4. `sweep_reversal`

**Scenario:** A cascade of short liquidations causes a sharp price spike downwards (a "sweep"), which is then followed by a quick reversal upwards.

```json
{
  "direction": "long",
  "conf_calibrated": 0.88,
  "reasons": ["liquidation_sweep_down", "reversal_after_sweep"],
  "strategy_id": "sweep_reversal"
}
```
