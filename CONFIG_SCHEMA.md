# Config Schema

This document outlines the configuration parameters for the trading system, organized by modules.

## Ensemble

| Parameter | Type | Description |
|---|---|---|
| `enabled` | boolean | If true, the ensemble feature is activated. |
| `strategies` | list[str] | A list of strategy identifiers to be used in the ensemble. |
| `weights` | dict | A dictionary defining the weights for each strategy. It can have a `default` set of weights and other profiles like `trend` and `mr` (mean-reversion). |
| `vote_threshold` | float | The minimum weighted confidence sum required to trigger a trading decision. |
| `veto` | dict | A set of conditions that can veto a decision, such as `breakout_requires_book_flip` and `mr_requires_band_pierce`. |

## Correlation

| Parameter | Type | Description |
|---|---|---|
| `lookbacks` | list[str] | A list of time periods (e.g., "1h", "4h") to look back when calculating correlations. |
| `refresh_min` | int | The frequency in minutes at which correlation data is refreshed. |
| `threshold` | float | The correlation coefficient threshold for grouping symbols into clusters. |
| `hysteresis_hits` | int | The number of consecutive times a correlation change must be observed before updating the cluster state. |

## Portfolio

| Parameter | Type | Description |
|---|---|---|
| `max_risk_per_symbol` | float | The maximum percentage of equity to risk on a single symbol. |
| `max_positions_per_symbol` | int | The maximum number of concurrent positions allowed for a single symbol. |
| `max_positions_total` | int | The total maximum number of positions that can be open at any time. |
| `max_cluster_risk` | float | The maximum risk exposure per cluster, with long and short positions tracked separately. |
| `max_net_long_risk` | float | The maximum total risk for all long positions. |
| `max_net_short_risk` | float | The maximum total risk for all short positions. |
| `margin_cap_pct` | float | The maximum percentage of equity that can be used as margin. |

## Brakes

| Parameter | Type | Description |
|---|---|---|
| `min_spacing_sec_same_cluster` | int | The minimum time in seconds required between trades in the same cluster. |
| `daily_loss_soft_pct` | float | The percentage of daily loss that triggers a soft stop (e.g., a cooldown period). |
| `daily_loss_hard_pct` | float | The percentage of daily loss that triggers a hard stop (e.g., halting trading for the day). |
| `cooldown_after_soft_min` | int | The duration in minutes of the cooldown period after a soft loss limit is hit. |
| `cooldown_after_streak_symbol` | dict | Defines a cooldown triggered by a losing streak on a specific symbol, with `losses` (number of consecutive losses) and `minutes` (cooldown duration). |
| `cooldown_after_streak_global` | dict | Defines a cooldown triggered by a global losing streak, with `losses` and `minutes`. |

## Volatility Risk Scale

| Parameter | Type | Description |
|---|---|---|
| `atr_pct_window` | int | The lookback window for calculating the Average True Range (ATR) percentile. |
| `low_vol_pct` | int | The percentile below which volatility is considered low. |
| `high_vol_pct` | int | The percentile above which volatility is considered high. |
| `low_vol_boost` | float | The factor by which position size is boosted in a low-volatility environment. |
| `high_vol_cut` | float | The factor by which position size is cut in a high-volatility environment. |