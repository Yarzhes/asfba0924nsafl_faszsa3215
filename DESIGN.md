# Ultra-Signals System Design

This document outlines the complete system design for the Ultra-Signals project.

## 1. Repository & File Responsibilities

This section mirrors the final file layout and the responsibility of each module.

```
ultra_signals/
  ├─ apps/
  │  ├─ realtime_runner.py        # CLI to run live engine (no exchange keys needed for public WS)
  │  └─ backtest_cli.py           # CLI to run backtests/walk-forward
  ├─ core/
  │  ├─ config.py                 # Load/validate settings.yaml + .env; pydantic models
  │  ├─ timeutils.py              # Timezones, funding windows, session clocks
  │  ├─ mathutils.py              # Common numerics; rolling windows; z-scores
  │  └─ feature_store.py          # Typed store for OHLCV, trades, book, derived features
  ├─ data/
  │  ├─ binance_ws.py             # WS client: kline, aggTrade, bookTicker/partial depth, markPrice, forceOrder
  │  ├─ rest_clients.py           # Funding history, misc REST (rate-limited)
  │  └─ oi_providers/
  │     ├─ base.py                 # Abstract OI/liquidations provider interface
  │     ├─ coinglass.py            # Implementation (optional; feature-flag)
  │     └─ coinalyze.py            # Implementation (optional; feature-flag)
  ├─ features/
  │  ├─ trend.py                  # EMA stack, Supertrend, Donchian, structure (HH/HL/LH/LL)
  │  ├─ momentum.py               # RSI, Stoch RSI, MACD
  │  ├─ volatility.py             # ATR, BB, squeeze detection
  │  ├─ volume_flow.py            # OBV, volume z-score, VWAP + bands, CVD
  │  ├─ orderbook.py              # Depth sums, imbalance, slope, spread, slippage estimator
  │  ├─ derivatives.py            # Funding snapshot/trail, OI deltas, liquidation pulse
  │  └─ regime.py                 # ADX, volatility buckets, trend vs mean-revert signal
  ├─ engine/
  │  ├─ scoring.py                # Component scores [-1,+1], weights, confidence transform
  │  ├─ risk_filters.py           # Spread/liquidity gates, regime checks, funding window rules
  │  ├─ entries_exits.py          # Entry zone logic, SL, TP1/TP2/trailing; lifecycle state machine
  │  ├─ sizing.py                 # Position sizing by risk %, leverage guardrails
  │  └─ selectors.py              # Symbol/timeframe selection, dynamic universe updates
  ├─ transport/
  │  └─ telegram.py               # Message formatter + sender, rate limiting
  ├─ backtest/
  │  ├─ event_runner.py           # Event-driven simulator; same engine API as realtime
  │  ├─ slippage.py               # Book/ATR-based slippage models
  │  ├─ metrics.py                # PF, hit rate, DD, AAE, MCC; CSV/JSON reports
  │  └─ walkforward.py            # Purged/embargoed walk-forward controller
  └─ tests/                      # Unit/integration test specs (Architect lists them)

settings.yaml               # Project config (see schema)
.env.example                # Env variable keys (no secrets)
requirements.txt            # Libs list (Architect proposes; Code pins versions)
README.md                   # Runbook & quickstart
```

---

## 2. Scoring Model

The core of the engine is a scoring model that synthesizes various market features into a single, actionable signal.

### 2.1. `derivatives_score`

This is a new composite score that aggregates signals from derivatives markets to provide a more nuanced view of market conviction. It is composed of three sub-features:

1.  **Funding Rate Score:** Captures the direction and intensity of funding payments, indicating positioning bias.
2.  **Open Interest (OI) Delta Score:** Measures the net change in open interest, signaling new capital flows.
3.  **Liquidation Pulse Score:** Identifies clusters of liquidations that can signal trend continuation or exhaustion.

**Composition:**

The `derivatives_score` is a weighted average of its components. The weights are configurable and are designed to balance the influence of each sub-feature.

*   `derivatives_score` = (`funding_score` * `w_funding`) + (`oi_delta_score` * `w_oi`) + (`liquidation_pulse_score` * `w_liquidation`)

*Default Weights:*
*   `w_funding`: 0.4
*   `w_oi`: 0.4
*   `w_liquidation`: 0.2

## 3. Regime-Based Profile Selection

The system adapts its behavior by selecting a scoring weight profile that matches the current market regime, as determined by the **Regime Classifier** (`S4-T4-Regime-Classifier-Spec.md`).

### 3.1. Logic Flow

1.  The **Regime Classifier** outputs the current market state (e.g., `{mode: trend, vol_bucket: high}`).
2.  The engine uses this output to look up the corresponding weighting profile from a predefined set (`S4-T5-Regime-Weights-Spec.md`).
3.  The selected profile's weights are then applied to the feature scores to calculate the final, regime-aware signal.

### 3.2. Mapping Regimes to Profiles

| Regime Classifier Output | Selected Profile | Rationale |
| :--- | :--- | :--- |
| `{mode: trend}` | `trend_following` | Prioritizes trend and momentum features. |
| `{mode: mean_revert}` | `mean_reversion` | Focuses on overbought/oversold signals. |
| `{vol_bucket: high}` | `volatility_breakout` | Weights momentum and volatility for breakouts. |
| *Otherwise* | `default` | A balanced, baseline profile is used. |


## 4. Synthetic Examples

The following examples illustrate the end-to-end scoring process.

### Example 1: Strong Bullish Trend

*   **Inputs:**
    *   **Market:** Sharp uptrend, price breaking previous highs.
    *   **Derivatives:** Funding is positive and rising, OI is increasing steadily, no significant liquidations.
    *   `funding_score`: 0.8
    *   `oi_delta_score`: 0.7
    *   `liquidation_pulse_score`: 0.0
*   **Regime Classification:**
    *   `ADX` > 25 -> `{mode: trend}`
*   **Profile Selection:**
    *   The `trend_following` profile is selected.
*   **Score Calculation:**
    1.  **`derivatives_score`**: (0.8 * 0.4) + (0.7 * 0.4) + (0.0 * 0.2) = 0.32 + 0.28 = **0.60**
    2.  **Other Scores**: `trend`=0.9, `momentum`=0.7, `volatility`=0.2
    3.  **Final Signal**: (0.9 * 0.6) + (0.7 * 0.3) + (0.2 * 0.1) + (0.60 * **0.1**) = 0.54 + 0.21 + 0.02 + 0.06 = **0.83**
        *Note: A 10% weight is allocated to the derivatives score in the final signal.*

### Example 2: Range-Bound with Mean Reversion

*   **Inputs:**
    *   **Market:** Price oscillating between clear support and resistance.
    *   **Derivatives:** Funding is flat, OI is churning with no clear direction, a small liquidation pulse was detected as price hit resistance.
    *   `funding_score`: 0.1
    *   `oi_delta_score`: -0.2
    *   `liquidation_pulse_score`: -0.5 (indicating potential exhaustion)
*   **Regime Classification:**
    *   `ADX` < 20 -> `{mode: mean_revert}`
*   **Profile Selection:**
    *   The `mean_reversion` profile is selected.
*   **Score Calculation:**
    1.  **`derivatives_score`**: (0.1 * 0.4) + (-0.2 * 0.4) + (-0.5 * 0.2) = 0.04 - 0.08 - 0.10 = **-0.14**
    2.  **Other Scores**: `trend`=0.1, `momentum`=-0.8, `volatility`=0.6
    3.  **Final Signal**: (0.1 * 0.0) + (-0.8 * 0.6) + (0.6 * 0.4) + (-0.14 * **0.1**) = 0.0 - 0.48 + 0.24 - 0.014 = **-0.254**

### Example 3: High Volatility Breakout

*   **Inputs:**
    *   **Market:** Price compressing into a tight range (squeeze), then a sudden surge in volume and price upward.
    *   **Derivatives:** Funding flips strongly positive, OI spikes, short liquidations cascade.
    *   `funding_score`: 0.9
    *   `oi_delta_score`: 0.9
    *   `liquidation_pulse_score`: 0.7 (strong continuation signal)
*   **Regime Classification:**
    *   `ATR Percentile` > 70% -> `{vol_bucket: high}`
*   **Profile Selection:**
    *   The `volatility_breakout` profile is selected.
*   **Score Calculation:**
    1.  **`derivatives_score`**: (0.9 * 0.4) + (0.9 * 0.4) + (0.7 * 0.2) = 0.36 + 0.36 + 0.14 = **0.86**
    2.  **Other Scores**: `trend`=0.5, `momentum`=0.9, `volatility`=0.8
    3.  **Final Signal**: (0.5 * 0.2) + (0.9 * 0.5) + (0.8 * 0.3) + (0.86 * **0.1**) = 0.10 + 0.45 + 0.24 + 0.086 = **0.876**

---
*This file serves as the initial design artifact. After this step, I will create the actual directory structure by placing a `README.md` in each folder.*