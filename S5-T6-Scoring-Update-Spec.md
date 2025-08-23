# S5-T6: Scoring Engine Update Specification

## 1. Introduction

This document defines the confluence rules for integrating the new VWAP, Order Book (OB), and Relative Strength (RS) features into the signal scoring engine. The goal is to create a robust scoring model that leverages confirmation from these independent factors to identify high-probability trade setups.

## 2. Scoring Confluence Rules

This section details the specific rules for scoring trade signals based on different market scenarios. Each rule is designed to look for a confluence of events, where multiple indicators align to support a single directional bias.

### 2.1. Trend-Pullback Scenario

*   **Condition:** Price is in an established trend and pulls back towards a dynamic support/resistance level.
*   **Rule:** Assign a high score to an entry signal if the price touches or enters the VWAP standard deviation bands **AND** there is supportive evidence from the order book (e.g., high liquidity on the bands, order book imbalance favoring the trend direction).

### 2.2. Breakout Scenario

*   **Condition:** Price breaks out of a consolidation range or a key technical level.
*   **Rule:** Assign a high score to a breakout signal only if it is confirmed by a **book-flip** in the direction of the breakout **AND** a positive **CVD (Cumulative Volume Delta) slope**, indicating aggressive market participation.

### 2.3. Relative Strength (RS) Gate

*   **Condition:** A valid trade signal (either pullback or breakout) is generated.
*   **Rule:** The signal's score is significantly boosted **only if** the asset is currently in the `Top-K Longs` list (for a buy signal) or the `Bottom-K Shorts` list (for a sell signal) from the dynamic universe. Signals for assets not on these lists should receive a lower score or be filtered out entirely.

## 3. Worked Examples

This section provides hypothetical examples to illustrate how the scoring rules are applied in practice.

### 3.1. Example 1: High-Score Trend-Pullback (Long)

*   **Scenario:** Asset XYZ is in a clear uptrend. Price pulls back to the area between the Session VWAP and the first lower standard deviation band. The order book shows a significant increase in bid liquidity right at the VWAP level, and the Order Book Imbalance (OBI) is consistently above 0.6.
*   **Scoring:**
    *   Price touches VWAP band: **+1 point**
    *   Supportive order book liquidity: **+1 point**
    *   OBI favors trend: **+1 point**
    *   **Total Score Modifier: +3 (High Confidence)**

### 3.2. Example 2: High-Score Breakout (Long)

*   **Scenario:** Asset ABC has been consolidating in a tight range. It then breaks above the range high. Simultaneously, the order book, which was previously bid-heavy, flips to ask-heaviness (a "book-flip"), and the CVD slope turns sharply positive.
*   **Scoring:**
    *   Breakout confirmed by book-flip: **+2 points**
    *   Aggressive buying confirmed by positive CVD slope: **+1 point**
    *   **Total Score Modifier: +3 (High Confidence)**

### 3.3. Example 3: Low-Score Signal due to RS Filter

*   **Scenario:** Asset PQR generates a valid trend-pullback buy signal that meets all the criteria in Example 1. However, PQR is not listed in the `Top-K Longs` list.
*   **Scoring:**
    *   Base score for pullback: +3 points
    *   RS Gate: Asset not in `Top-K Longs`. **Score Multiplier: 0.5 (Reduced Confidence)**
    *   **Final Score: 1.5 (Low Confidence/Filtered)**
