# S4-T5: Regime-Aware Weights Profile Specification

## 1. Overview

This document defines a set of named, regime-aware weighting profiles for the trading engine. These profiles are designed to adapt the engine's scoring logic to different market regimes as identified by the `S4-T4-Regime-Classifier-Spec`. Each profile consists of a vector of feature weights and a corresponding entry template.

The feature weights defined in `settings.yaml` (`trend: 0.5`, `momentum: 0.5`, `volatility: 0.0`) serve as a baseline but will be superseded by these regime-specific profiles. Existing signal thresholds for entry and exit (`enter: 0.6`, `exit: 0.4`) remain unchanged unless explicitly stated otherwise in a specific profile's strategy.

## 2. Weighting Profiles

The following profiles provide distinct weighting strategies tailored to specific market conditions.

### 2.1. Trend Following Profile (`trend_following`)

This profile is optimized for markets exhibiting strong directional character, as identified when the regime classifier returns `{mode: trend}`.

*   **Entry Template:** "Enter on continuation or shallow pullback."
*   **Rationale:** In a trending market, momentum and trend-confirming signals are paramount. The highest weight is given to the `trend` component to capitalize on sustained directional moves. `Momentum` is also weighted significantly to ensure entries align with the current thrust. `Volatility` is given a minimal weight, primarily to act as a filter against excessively erratic conditions rather than as a primary signal driver.

*   **Feature Weights:**
    *   `trend`: 0.6
    *   `momentum`: 0.3
    *   `volatility`: 0.1
    *   **Total:** 1.0

### 2.2. Mean Reversion Profile (`mean_reversion`)

This profile is designed for oscillating, non-trending markets where prices tend to revert to a 'mean', as identified when the regime classifier returns `{mode: mean_revert}`.

*   **Entry Template:** "Enter on reversion to the mean."
*   **Rationale:** In a mean-reverting environment, the concept of a persistent `trend` is counter-productive. Therefore, its weight is set to zero. The strategy relies on identifying over-extended moves that are likely to snap back. `Momentum` indicators (like RSI) are critical for spotting these overbought/oversold conditions. `Volatility` (e.g., using Bollinger Bands) helps identify the boundaries of the expected price range, making it a key component for timing entries.

*   **Feature Weights:**
    *   `trend`: 0.0
    *   `momentum`: 0.6
    *   `volatility`: 0.4
    *   **Total:** 1.0

### 2.3. Volatility Breakout Profile (`volatility_breakout`)

This profile is intended for periods of high volatility where the market is likely to break out of a consolidation range, as identified when the regime classifier returns `{vol_bucket: high}`.

*   **Entry Template:** "Enter on breakout from range."
*   **Rationale:** This profile activates when volatility is high, suggesting a significant price move is imminent. The primary driver for a breakout is a sudden burst of `momentum`, hence its high weighting. `Volatility` itself is used to confirm the explosive condition. The `trend` component has a moderate weight, acting as a filter to ensure the breakout aligns with any emerging directional bias.

*   **Feature Weights:**
    *   `trend`: 0.2
    *   `momentum`: 0.5
    *   `volatility`: 0.3
    *   **Total:** 1.0

## 3. Profile Application Logic

The trading engine will dynamically select the appropriate weighting profile based on the `current_regime` provided by the classifier. This ensures that the signal scoring logic is always aligned with the most current assessment of the market's state.
