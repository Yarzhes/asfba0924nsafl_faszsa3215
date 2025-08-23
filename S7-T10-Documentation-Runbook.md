# S7-T10: Documentation & Operator Runbook

This document contains the `README` additions for system configuration and the troubleshooting runbook for operators.

---

## 1. README Additions: Configuration Guide

### Configuring the Ensemble and Portfolio Risk Systems

The following settings control the behavior of the signal ensemble and the portfolio-level risk management systems. These are typically located in your `settings.yaml` file under the `engine` configuration block.

#### Ensemble Configuration

The `ensemble` section configures how individual strategy signals are combined into a final trade decision.

**Example Configuration:**

```yaml
engine:
  ensemble:
    min_agreement_pct: 60       # Minimum percentage of strategies that must agree for a signal to be valid. (0-100)
    weighting_strategy: "equal" # 'equal' or 'confidence_weighted'.
    confidence_scale_factor: 1.2 # Multiplier for confidence scores when weighting_strategy is 'confidence_weighted'.
```

*   `min_agreement_pct`: This is a critical parameter for controlling signal consensus. A higher value means more strategies must agree, leading to fewer but potentially higher-quality signals.
*   `weighting_strategy`: Determines how to weigh the participating signals.
    *   `equal`: All firing strategies are given the same weight.
    *   `confidence_weighted`: Each strategy's vote is multiplied by its calibrated confidence score.
*   `confidence_scale_factor`: A tuning parameter to amplify or dampen the effect of confidence scores when using `confidence_weighted`.

#### Portfolio Risk Controls

The `portfolio_risk` section defines global risk controls that can veto or scale trades based on overall portfolio state.

**Example Configuration:**

```yaml
engine:
  portfolio_risk:
    max_open_positions: 10              # Absolute maximum number of open positions allowed.
    max_exposure_usd: 50000             # Maximum total notional exposure in USD.
    trade_spacing_seconds: 300          # Minimum time in seconds between two new trades.
    correlation_veto_threshold: 0.85    # Veto a new trade if its correlation to an existing position exceeds this value.
    volatility_scaling:
      enabled: true
      target_daily_vol_pct: 1.5       # Target daily portfolio volatility percentage. Position sizes will be scaled down if portfolio vol exceeds this.
```

*   `max_open_positions`: Prevents the system from accumulating too many small positions.
*   `max_exposure_usd`: A hard cap on the total market exposure.
*   `trade_spacing_seconds`: Acts as a circuit breaker to prevent rapid, successive trades in volatile conditions.
*   `correlation_veto_threshold`: A filter to improve diversification. If a new trade is highly correlated with an existing one, it will be blocked.
*   `volatility_scaling`: An adaptive sizing mechanism. When enabled, it adjusts position sizes to maintain a stable target portfolio volatility.

---

## 2. Troubleshooting Runbook for Operators

This runbook provides solutions to common operational issues.

### Problem: High Slippage or Poor Fills

*   **Likely Cause:** The `slippage_model` in the backtest or the live execution cost model is misconfigured and doesn't reflect real market conditions (e.g., assuming too much available liquidity).
*   **Solution:**
    1.  Review the live execution logs to compare expected vs. actual fill prices.
    2.  Check the `cost_model_spec` and `slippage_models` in the configuration.
    3.  Consider using a more conservative slippage model, such as one based on a percentage of the ATR (Average True Range) instead of a fixed basis point cost.

### Problem: System is not sending Telegram notifications

*   **Likely Cause:** The Telegram API token or chat ID is incorrect, or the transport is disabled.
*   **Solution:**
    1.  Verify that the `transport.telegram.enabled` flag is set to `true` in `settings.yaml`.
    2.  Confirm that the `TELEGRAM_API_TOKEN` and `TELEGRAM_CHAT_ID` environment variables are correctly set and exported in the execution environment.
    3.  Check for any firewall rules that might be blocking outbound HTTPS requests to `api.telegram.org`.

### Problem: Why are there no trades?

This is a common and important question. Go through this checklist in order.

*   **Symptom -> Cause -> Investigation Step**

1.  **Restrictive Ensemble Rules?**
    *   **Cause:** The `min_agreement_pct` for the ensemble is set too high, so no consensus is ever reached.
    *   **Investigation:** Check the logs for messages like "Ensemble vote failed: agreement XX% is below threshold YY%." Lower the `min_agreement_pct` in `settings.yaml` if it's excessively high (e.g., > 80%).

2.  **Risk Controls Vetoing Trades?**
    *   **Cause:** One of the portfolio risk filters is blocking the trade. This is often the intended behavior of the system.
    *   **Investigation:** Check the logs for specific veto messages:
        *   `"Trade vetoed: max open positions reached"` -> Check `portfolio_risk.max_open_positions`.
        *   `"Trade vetoed: max exposure reached"` -> Check `portfolio_risk.max_exposure_usd`.
        *   `"Trade vetoed: trade spacing active"` -> A trade occurred recently. Check `portfolio_risk.trade_spacing_seconds`.
        *   `"Trade vetoed: high correlation with existing position"` -> The proposed trade is too similar to something already in the portfolio. Check `portfolio_risk.correlation_veto_threshold`.

3.  **Low Signal Confidence?**
    *   **Cause:** If using `confidence_weighted` ensemble, the strategies might be producing signals, but their confidence scores are too low to pass a downstream threshold (if any).
    *   **Investigation:** Examine the strategy-level logs to see the confidence scores being generated for each signal. Check if there's a minimum confidence filter that is not being met.

4.  **No Base Signals Firing?**
    *   **Cause:** The underlying strategies themselves are not generating any entry or exit signals. This could be due to market conditions (e.g., low volatility, no clear trends) or a bug in a specific feature.
    *   **Investigation:**
        1.  Check the individual strategy logs to confirm if they are analyzing data.
        2.  Look for any errors in the feature calculation modules (e.g., `features/trend.py`, `features/volatility.py`).
        3.  Confirm that the market data feed is active and providing data to the system.

5.  **Is the System Running?**
    *   **Cause:** The `realtime_runner.py` script may have crashed or was never started.
    *   **Investigation:**
        1.  Run `ps aux | grep realtime_runner.py` to check if the process is active.
        2.  Check the main application log file for any fatal error messages or stack traces that would indicate a crash.