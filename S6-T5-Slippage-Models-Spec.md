# S6-T5: Slippage Models Specification

This document specifies two distinct slippage models to simulate the market impact of trades within the backtester. Slippage is the difference between the expected price of a trade and the price at which the trade is actually executed.

## 1. Model Selection and Configuration

A user can select and configure the desired slippage model via the backtest configuration. The `slippage` object will contain the `model` name and its specific parameters. If the `slippage` section is omitted, no slippage will be applied.

### Example Configuration:

**ATR-Based Model:**
```yaml
slippage:
  model: 'atr'
  # Slippage is ATR * multiplier
  multiplier: 0.1
```

**Book-Proxy Model:**
```yaml
slippage:
  model: 'book_proxy'
  # Proportional impact based on trade size vs. bar volume
  impact_factor: 0.5
  # Exponent to control the non-linearity of the impact
  exponent: 1.0
```

---

## 2. ATR-Based Slippage Model

This model calculates slippage as a fixed fraction of the Average True Range (ATR) at the time of the trade. It assumes that slippage is primarily a function of recent market volatility.

### 2.1. Formula

The slippage per unit of asset is calculated and then applied to the fill price.

*   `SlippagePerUnit = ATR * Multiplier`

Where:
*   `ATR`: The Average True Range value for the bar during which the trade occurs.
*   `Multiplier`: A configurable parameter representing the fraction of ATR to apply as slippage.

The slippage is always unfavorable:
*   For **buy** orders, the effective price is increased: `FillPrice = EntryPrice + SlippagePerUnit`
*   For **sell** orders, the effective price is decreased: `FillPrice = ExitPrice - SlippagePerUnit`

### 2.2. Input Parameters

| Parameter  | Type    | Description                                       |
|------------|---------|---------------------------------------------------|
| `Entry/Exit Price` | float   | The theoretical execution price before slippage. |
| `ATR`        | float   | The ATR value for the current bar.                |
| `Multiplier` | float   | Configurable slippage factor.                     |

### 2.3. Test Vector

**Scenario:**
*   A **long entry** (buy) order is executed.
*   **Entry Price:** 35,000 USDT
*   **ATR:** 150 USDT
*   **Configuration:** `slippage: { model: 'atr', multiplier: 0.2 }`

**Calculation:**
1.  `SlippagePerUnit = 150 * 0.2 = 30 USDT`
2.  `FillPrice = 35,000 (Entry) + 30 = 35,030 USDT`

**Result:** The executed fill price is 35,030 USDT.

---

## 3. Book-Proxy Slippage Model

This model simulates slippage by considering the trade size relative to the available liquidity, which is proxied by the bar's volume and price range. It models the non-linear impact where larger trades experience disproportionately more slippage.

### 3.1. Formula

The slippage per unit is calculated based on the ratio of the trade's volume to the bar's volume, scaled by the bar's price range.

*   `SlippagePerUnit = (TradeSize / BarVolume) ^ Exponent * (BarHigh - BarLow) * ImpactFactor`

Where:
*   `TradeSize`: The size of the trade in the base asset (e.g., in BTC).
*   `BarVolume`: The total traded volume for the bar in the base asset.
*   `BarHigh`: The high price of the bar.
*   `BarLow`: The low price of the bar.
*   `ImpactFactor`: A configurable multiplier to scale the overall slippage effect.
*   `Exponent`: A configurable exponent to model the non-linearity of market impact (e.g., `1.0` for linear, `>1.0` for increasing impact).

The slippage is always unfavorable:
*   For **buy** orders: `FillPrice = EntryPrice + SlippagePerUnit`
*   For **sell** orders: `FillPrice = ExitPrice - SlippagePerUnit`

### 3.2. Input Parameters

| Parameter      | Type  | Description                                        |
|----------------|-------|----------------------------------------------------|
| `Entry/Exit Price` | float | The theoretical execution price before slippage.   |
| `TradeSize`      | float | The size of the trade in the base asset.           |
| `BarVolume`      | float | Total volume traded during the bar.                |
| `BarHigh`        | float | The high price of the bar.                         |
| `BarLow`         | float | The low price of the bar.                          |
| `ImpactFactor`   | float | Configurable scaling factor for the impact.        |
| `Exponent`       | float | Configurable exponent for non-linear impact.       |

### 3.3. Test Vector

**Scenario:**
*   A **short entry** (sell) order is executed.
*   **Entry Price:** 50,000 USDT
*   **Trade Size:** 10 BTC
*   **Bar Data:**
    *   `High`: 50,100 USDT
    *   `Low`: 49,900 USDT
    *   `Volume`: 500 BTC
*   **Configuration:** `slippage: { model: 'book_proxy', impact_factor: 0.5, exponent: 1.0 }`

**Calculation:**
1.  `PriceRange = 50,100 - 49,900 = 200 USDT`
2.  `VolumeRatio = 10 / 500 = 0.02`
3.  `SlippagePerUnit = (0.02)^1.0 * 200 * 0.5 = 0.02 * 100 = 2.0 USDT`
4.  `FillPrice = 50,000 (Entry) - 2.0 = 49,998 USDT`

**Result:** The executed fill price is 49,998 USDT.