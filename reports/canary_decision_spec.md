
# Canary Decision Specification

This document outlines the ordered checklist the canary system uses to approve or deny a signal.

1.  **Data Health Gates (S39)**:
    *   **Time Sync**: Clock skew must be < `250` ms.
    *   **Staleness**: Market data heartbeat must be < `20` seconds old.
    *   **Missing Bars**: No more than `?` consecutive bars missing.
    *   **Book Quality**: Spread must be < `5.0`%.

2.  **Alpha Candidates & Meta-Scorer (S2, S11, S31, S48)**:
    *   **Alpha Emitters**: A set of alphas (e.g., `breakout_v2`, `rsi_extreme`) must fire based on the current regime profile.
    *   **Ensemble Vote**: A minimum of `?` votes are required for a trend decision.
    *   **Meta-Scorer `p_win`**: The ML model's calibrated probability of winning (`p_win`) must be >= `?`.

3.  **Multi-Timeframe (MTF) Confirmation (S30)**:
    *   **Rule**: The signal on the primary timeframe (e.g., 5m) must align with the trend/bias of a higher timeframe (e.g., 15m).
    *   **Agreement**: "Agree" means the 15m chart is also in a trend regime and its EMA structure supports the 5m signal direction. Enabled via `mtf_confirm: true`.

4.  **Regime & Volatility Context (S61, S43, S52)**:
    *   **Minimum Confidence**: The probabilistic regime model (S61) must have a confidence > 0.6 (example value) in the current regime.
    *   **Volatility Forecast**: GARCH models (S52) must not predict an imminent volatility expansion that would invalidate the trade thesis.

5.  **Veto Stack (Hard Gates)**:
    *   **VPIN (S49)**: VPIN score must be below a threshold (e.g., 0.7) to avoid toxic flow.
    *   **Kyle's Lambda (S50)**: Market impact estimate must be low.
    *   **Funding/OI (S54)**: Signal is blocked if approaching a funding window (`?` mins) or if OI spikes suggest a squeeze.
    *   **Circuit Breaker (S65)**: Global or symbol-specific circuit breakers (e.g., from extreme losses) must be inactive.
    *   **Liquidity/Micro-Regime (S29)**: Spread must not be excessively wide (`?` bps).

6.  **Sizing & Risk Eligibility (S12, S32, S34, S37)**:
    *   The canary **checks** if a valid position size could be calculated.
    *   It confirms that adaptive stops/targets (S34/S37) can be determined, but places no orders.

7.  **Sniper & Rate Caps**:
    *   **Hourly Cap**: Total signals for the symbol < `30`.
    *   **Daily Cap**: Total signals for the symbol < `200`.
    *   *Note: These are set high in the canary profile to prevent premature blocking.*

8.  **Telegram Emission Rules**:
    *   **PRE Allowed**: A `PRE` message is sent if a signal passes ALL the gates above.
    *   **PRE Blocked**: A `BLOCKED` debug message is sent if `send_blocked_signals_in_canary` is true and the signal is vetoed.
    *   **No Emission**: Nothing is sent if there are no initial alpha candidates or if data quality gates fail at the very start.
