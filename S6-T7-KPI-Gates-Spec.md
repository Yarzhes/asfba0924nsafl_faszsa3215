# S6-T7: KPI Gates and Thresholds Specification

## 1. Introduction

This document defines the official Key Performance Indicators (KPIs), minimum performance thresholds, and reporting standards used to evaluate all strategy backtests and walk-forward analyses. A strategy must pass all defined KPI gates to be considered viable for live deployment.

## 2. KPI Definitions

This section details the standard set of KPIs calculated for every performance report.

*   **Profit Factor (PF):** The ratio of gross profits to gross losses. It measures how many times the total profits exceed the total losses.
    *   *Formula:* `Gross Profit / |Gross Loss|`
*   **Hit Rate (Win Rate):** The percentage of trades that are profitable.
    *   *Formula:* `(Number of Winning Trades / Total Number of Trades) * 100`
*   **Average Risk:Reward Ratio (Avg R:R):** The ratio of the average profit on winning trades to the average loss on losing trades.
    *   *Formula:* `Average Win / |Average Loss|`
*   **Maximum Drawdown (Max DD):** The largest peak-to-trough decline in portfolio value, expressed as a percentage. It is a key measure of downside risk.
*   **Sharpe Ratio:** Measures the risk-adjusted return of the strategy. It is calculated by subtracting the risk-free rate from the portfolio's rate of return and dividing by the standard deviation of the portfolio's excess return. A higher Sharpe Ratio indicates better performance for the amount of risk taken.
*   **Sortino Ratio:** A variation of the Sharpe Ratio that only considers downside volatility (standard deviation of negative returns). It differentiates harmful volatility from total overall volatility.
*   **Matthews Correlation Coefficient (MCC):** A balanced measure of the quality of a binary classification. In trading, it considers true positives (correctly predicted wins), true negatives (correctly predicted non-trades/losses), false positives (incorrectly predicted wins), and false negatives (missed wins). It produces a value between -1 and +1, where +1 is a perfect prediction, 0 is random, and -1 is an inverse prediction.
*   **Average Adverse/Favorable Excursion (AAE/AFE):**
    *   **Maximum Adverse Excursion (MAE):** The largest unrealized loss a trade experiences before it is closed. Averaging this across all winning trades helps to evaluate the quality of stop-loss placement.
    *   **Maximum Favorable Excursion (MFE):** The largest unrealized profit a trade experiences before it is closed. Averaging this across all losing trades can indicate if profits are being given back.

## 3. KPI Gates and Minimum Thresholds

This section establishes the minimum "pass" threshold for each KPI. A strategy's failure to meet any of these thresholds will result in a "FAIL" status for the entire evaluation.

### KPI Summary Table

| KPI | Minimum Threshold | Rationale |
| --- | ------------------- | --------- |
| **Profit Factor (PF)** | > 1.2 | A value greater than 1.0 indicates profitability. A threshold of 1.2 is set to ensure the strategy overcomes transaction costs and potential slippage with a reasonable margin. |
| **Hit Rate (Win Rate)** | > 35% | While a high hit rate is not strictly necessary for profitability (if R:R is high), a rate below 35% can be psychologically difficult to trade and may indicate a flawed edge. |
| **Average Risk:Reward (R:R)** | > 1.5 | The average win should be significantly larger than the average loss. A minimum R:R of 1.5 ensures that wins are substantial enough to cover multiple losses, creating a positive expectancy. |
| **Maximum Drawdown (Max DD)** | < 20% | Excessive drawdowns can lead to investor panic and capital depletion. A threshold of 20% is set as an aggressive but acceptable level of risk for initial strategy validation. This may be adjusted based on asset class and volatility. |
| **Sharpe Ratio** | > 1.0 | A Sharpe Ratio greater than 1.0 is generally considered good, indicating that the strategy is generating excess returns relative to its volatility and the risk-free rate. |
| **Sortino Ratio** | > 1.5 | The Sortino ratio should be significantly higher than the Sharpe ratio, as it only penalizes downside risk. A value of 1.5 or higher demonstrates strong performance without excessive negative volatility. |
| **Matthews Corr. Coeff. (MCC)** | > 0.1 | The MCC provides a more balanced evaluation than simple accuracy. A value greater than 0.1 indicates a positive correlation between the predictions and actual outcomes, suggesting the model has *some* predictive power beyond random chance. |
| **AAE/AFE Analysis** | Manual Review | There is no strict numerical gate for AAE/AFE. These metrics will be used for manual, qualitative review to diagnose issues with stop-loss placement (AAE on wins) and take-profit levels (AFE on losses). |

## 4. Reporting Output Format

The final backtest report must include a dedicated section titled "Performance Summary & KPI Gates". This section will contain a table that clearly lists each KPI, its calculated value from the backtest, the minimum threshold, and a final "Pass/Fail" status for each gate.

### Example Report Table:

| KPI | Value | Threshold | Status |
| --- | ----- | --------- | ------ |
| Profit Factor | 1.45 | > 1.2 | PASS |
| Hit Rate | 42% | > 35% | PASS |
| Average R:R | 1.8 | > 1.5 | PASS |
| Maximum Drawdown | -15.3% | < 20% | PASS |
| Sharpe Ratio | 1.1 | > 1.0 | PASS |
| Sortino Ratio | 1.9 | > 1.5 | PASS |
| MCC | 0.15 | > 0.1 | PASS |
| **Overall Result**| - | - | **PASS** |

The report must conclude with a clear, final **Overall Result** of **PASS** or **FAIL**. A **FAIL** status for any single KPI gate will result in an overall **FAIL** for the strategy evaluation.
