# S5-T4: Relative Strength (RS) Technical Design Specification

## 1. Overview

This document outlines the technical design for the Relative Strength (RS) module. The module will calculate the relative strength of assets within a given universe, normalize the scores, and produce lists of top- and bottom-performing assets. This output will be used by the scoring engine and other downstream components.

## 2. Relative Strength Formula

The Relative Strength (RS) of an asset will be calculated as a weighted average of its percentage price change over multiple lookback periods. This approach allows us to capture both short-term and medium-term momentum.

### Formula

The RS for a single asset `i` is defined as:

`RS_i = (w_1 * PctChange_i,1) + (w_2 * PctChange_i,2) + ... + (w_n * PctChange_i,n)`

Where:
- `w_n`: The weight assigned to lookback period `n`. The sum of all weights must equal 1.
- `PctChange_i,n`: The percentage price change of asset `i` over the lookback period `n`.

`PctChange_i,n = (Price_i,t0 / Price_i,t-n) - 1`

Where:
- `Price_i,t0`: The current price of asset `i`.
- `Price_i,t-n`: The price of asset `i` at the start of the lookback period `n`.

### Lookback Periods and Weights

The following lookback periods and weights are configured for the initial implementation:

| Lookback Period | Weight (`w`) |
|-----------------|--------------|
| 1-Hour          | 0.4          |
| 4-Hour          | 0.6          |

## 3. Normalization

To make the raw RS scores comparable across the entire asset universe, we will normalize them using the Z-score. The Z-score indicates how many standard deviations an element is from the mean.

### Formula

The Z-score for an asset `i` is calculated as follows:

`Z_i = (RS_i - μ_RS) / σ_RS`

Where:
- `Z_i`: The normalized Z-score for asset `i`.
- `RS_i`: The raw Relative Strength score for asset `i`.
- `μ_RS`: The mean of all RS scores across the entire asset universe.
- `σ_RS`: The standard deviation of all RS scores across the entire asset universe.

## 4. Top-K / Bottom-K Lists

After normalizing the RS scores, all assets in the universe are ranked from highest to lowest based on their Z-scores. This ranking is used to generate two distinct lists: `Top-K Longs` and `Bottom-K Shorts`.

### List Generation Logic

1.  **Ranking:** All assets are sorted in descending order based on their normalized `Z_i` score.
2.  **Top-K Longs:** The top `K` assets with the highest Z-scores are selected for the "longs" list.
3.  **Bottom-K Shorts:** The bottom `K` assets with the lowest Z-scores are selected for the "shorts" list.
4.  The value of `K` will be configurable, with a default value of 10.

### Tie-Breaking

In the event of a tie in Z-scores, the following rules will be applied in order:

1.  **Lower Volatility:** The asset with lower price volatility over a 24-hour period will be given a higher rank.
2.  **Alphabetical Order:** If volatility is also equal, the assets will be sorted alphabetically by their symbol (e.g., 'BTCUSDT' comes before 'ETHUSDT').

## 5. Rebalance Cadence

The RS scores and the resulting Top-K / Bottom-K lists will be recalculated at a fixed interval. This ensures that the lists remain fresh and reflect the latest market movements.

- **Frequency:** The rebalance will occur every **15 minutes**.

This cadence is configurable and can be adjusted based on system performance and strategy requirements.

## 6. Output Contract

The RS module will output a JSON object containing the ranked lists of long and short candidates, along with metadata for the calculation cycle. This contract ensures that downstream consumers, such as the scoring engine, have a consistent and predictable data structure.

### JSON Schema

```json
{
  "timestamp": "2023-10-27T10:00:00Z",
  "k_value": 10,
  "top_k_longs": [
    {
      "rank": 1,
      "symbol": "ASSET-A",
      "z_score": 2.5
    },
    {
      "rank": 2,
      "symbol": "ASSET-B",
      "z_score": 2.1
    }
  ],
  "bottom_k_shorts": [
    {
      "rank": 99,
      "symbol": "ASSET-X",
      "z_score": -2.3
    },
    {
      "rank": 100,
      "symbol": "ASSET-Y",
      "z_score": -2.8
    }
  ]
}
```

### Field Descriptions

| Field             | Type    | Description                                                                 |
|-------------------|---------|-----------------------------------------------------------------------------|
| `timestamp`       | String  | ISO 8601 timestamp of when the calculation was performed.                   |
| `k_value`         | Integer | The number of assets included in the long and short lists.                  |
| `top_k_longs`     | Array   | An array of objects, each representing an asset with high relative strength.  |
| `bottom_k_shorts` | Array   | An array of objects, each representing an asset with low relative strength.   |
| `rank`            | Integer | The asset's rank within the entire universe (1 being the highest).          |
| `symbol`          | String  | The symbol of the asset.                                                    |
| `z_score`         | Float   | The normalized Z-score of the asset.                                        |
