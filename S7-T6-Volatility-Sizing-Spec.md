# S7-T6: Volatility-Scaled Position Sizing Specification

**Author:** Roo
**Date:** 2025-08-22
**Status:** Draft

## 1. Overview

This document specifies a dynamic position sizing model that adjusts the risk taken per trade based on market volatility. The primary goal is to scale risk inversely to volatility: taking larger risks in low-volatility environments and smaller risks in high-volatility environments. This is achieved by using the Average True Range (ATR) as a proxy for volatility and scaling the allocated risk based on its percentile rank over a lookback period.

## 2. Sizing Formula

The core of this model is a scaling factor, `VolatilityScale`, which modulates the maximum allowable risk per trade (`MaxRiskPerTrade`, defined in `S7-T4-Exposure-Caps-Policy.md`).

The adjusted risk for a given trade is calculated as follows:

```
AdjustedRisk = MaxRiskPerTrade * VolatilityScale
```

### 2.1. Volatility Scale Formula

The `VolatilityScale` is determined by the current ATR percentile (`ATR_Percentile`) over a specified lookback period. The formula is designed to produce a scaling factor between a defined `MinScale` and `MaxScale`.

The formula is as follows:

```
VolatilityScale = MinScale + (MaxScale - MinScale) * (1 - ATR_Percentile)
```

Where:
- **`ATR_Percentile`**: The percentile rank of the current ATR value over a lookback period. A value of `1.0` means the current ATR is the highest in the period, while `0.0` means it is the lowest.

## 3. Bounds and Parameters

The behavior of the formula is controlled by several configurable parameters.

### 3.1. Parameters

| Parameter | Description | Default Value | Rationale |
| --- | --- | --- | --- |
| `ATR_LookbackPeriod` | The number of periods (e.g., days) to use for calculating the ATR percentile. | `100` | A longer lookback period provides a more stable and robust measure of the volatility regime. |
| `MinScale` | The minimum scaling factor, applied when volatility is at its highest (`ATR_Percentile` = 1.0). | `0.5` | Prevents taking excessively small positions while still significantly reducing risk in high-volatility regimes. |
| `MaxScale` | The maximum scaling factor, applied when volatility is at its lowest (`ATR_Percentile` = 0.0). | `1.25` | Allows for a modest increase in risk-taking during exceptionally low-volatility periods to capitalize on potential opportunities. |
| `MaxRiskPerTrade`| The maximum risk (as a fraction of portfolio equity) allowed for a single trade before volatility scaling. | `0.02` (2%) | This is the baseline risk cap defined in `S7-T4-Exposure-Caps-Policy.md`. |

### 3.2. Bounds

- **`VolatilityScale`**: The output of the formula is naturally bounded between `MinScale` and `MaxScale`.
  - If `ATR_Percentile` = `1.0` (highest volatility), `VolatilityScale` = `0.5`.
  - If `ATR_Percentile` = `0.0` (lowest volatility), `VolatilityScale` = `1.25`.

## 4. Worked Examples

Let's assume the following parameters:
- `MaxRiskPerTrade` = `2%`
- `MinScale` = `0.5`
- `MaxScale` = `1.25`

### 4.1. Scenario 1: High Volatility

- **Context:** A major news event causes a massive spike in volatility.
- **Current ATR:** The current ATR is the highest it has been in the last 100 periods.
- **`ATR_Percentile`**: `1.0`

**Calculation:**
1.  **`VolatilityScale`** = `0.5 + (1.25 - 0.5) * (1 - 1.0)`
    = `0.5 + 0.75 * 0`
    = `0.5`
2.  **`AdjustedRisk`** = `2% * 0.5`
    = `1%`

**Result:** In this high-volatility scenario, the risk per trade is halved from `2%` to `1%` of the portfolio.

### 4.2. Scenario 2: Medium Volatility

- **Context:** The market is behaving normally, with no significant volatility spikes or troughs.
- **Current ATR:** The current ATR is around the median value for the lookback period.
- **`ATR_Percentile`**: `0.5`

**Calculation:**
1.  **`VolatilityScale`** = `0.5 + (1.25 - 0.5) * (1 - 0.5)`
    = `0.5 + 0.75 * 0.5`
    = `0.5 + 0.375`
    = `0.875`
2.  **`AdjustedRisk`** = `2% * 0.875`
    = `1.75%`

**Result:** In a normal volatility environment, the risk per trade is slightly reduced to `1.75%` of the portfolio.

### 4.3. Scenario 3: Low Volatility

- **Context:** The market is unusually quiet and trading in a tight range.
- **Current ATR:** The current ATR is the lowest it has been in the last 100 periods.
- **`ATR_Percentile`**: `0.0`

**Calculation:**
1.  **`VolatilityScale`** = `0.5 + (1.25 - 0.5) * (1 - 0.0)`
    = `0.5 + 0.75 * 1`
    = `1.25`
2.  **`AdjustedRisk`** = `2% * 1.25`
    = `2.5%`

**Result:** In this low-volatility scenario, the risk per trade is increased by 25% to `2.5%` of the portfolio, allowing for larger positions to be taken.
