# Sprint 5 Test Plan (S5-T8)

**Objective:** This document outlines the testing strategy for the new features developed in Sprint 5. The goal is to validate the correctness, robustness, and performance of the new VWAP, Order Book, Relative Strength, scoring, and filtering logic defined in tasks S5-T1 through S5-T7.

---

## 1. Unit Test Plan

This section details the unit tests for individual components and functions.

### 1.1. VWAP Calculations
- **[ ] VWAP - Rolling:**
  - **Property:** Correctly calculates the rolling VWAP over a specified window.
  - **Property:** Handles empty or incomplete data gracefully.
  - **Property:** Output matches a trusted reference calculation.
- **[ ] VWAP - Session:**
  - **Property:** Correctly calculates VWAP anchored to the start of a session.
  - **Property:** Resets correctly at the start of a new session.
- **[ ] VWAP - Anchored:**
  - **Property:** Correctly calculates VWAP from a specified anchor point (e.g., a high/low).

### 1.2. Order Book Feature Calculations
- **[ ] Order Book Imbalance:**
  - **Property:** Correctly calculates the imbalance ratio.
  - **Property:** Handles edge cases (e.g., zero depth on one side).
- **[ ] Book-Flip:**
  - **Property:** Correctly identifies a book-flip event based on the specified logic.
  - **Property:** Differentiates between transient and significant flips.
- **[ ] Cumulative Volume Delta (CVD):**
  - **Property:** Correctly calculates the cumulative delta of market orders.
- **[ ] Slippage:**
  - **Property:** Correctly estimates slippage for a given order size.

### 1.3. Relative Strength (RS) Calculation
- **[ ] RS Score:**
  - **Property:** Correctly calculates the RS score based on the defined lookback period.
- **[ ] Z-Score:**
  - **Property:** Correctly calculates the z-score of the RS score.
  - **Property:** Handles periods of low volatility without errors.

---

## 2. Integration Test Plan

This section defines test scenarios that verify the interaction between different components.

### 2.1. Scoring and Feature Integration
- **[ ] Scenario: Trend-Pullback Signal with VWAP/OB Confirmation**
  - **Given:** A trend-pullback signal is generated.
  - **When:** The asset is trading above the session VWAP and the order book shows strong bid-side imbalance.
  - **Then:** The signal's final score is significantly boosted.
- **[ ] Scenario: Breakout Signal without Book-Flip Confirmation**
  - **Given:** A breakout signal is generated.
  - **When:** A book-flip event has *not* occurred within the confirmation window.
  - **Then:** The signal is blocked by the "no-flip" risk filter.
- **[ ] Scenario: Signal for Asset Outside RS Universe**
  - **Given:** A signal is generated for an asset.
  - **When:** The asset is not in the `Top-K/Bottom-K` RS lists.
  - **Then:** The signal's final score is penalized as per the dynamic universe policy.

---

## 3. Live Soak Test Plan

This section outlines the plan for a live soak test to monitor a near-production environment.

### 3.1. Test Parameters
- **[ ] Duration:** 60 minutes.
- **[ ] Environment:** Live market data feed, running in a shadow/paper trading mode.
- **[ ] Active Features:** All new Sprint 5 features enabled.

### 3.2. Monitoring
- **[ ] Data to Collect:**
  - **Logs:** System logs, focusing on errors and warnings.
  - **Performance:** CPU and Memory usage of the application process.
  - **Signal Metrics:**
    - Total number of signals generated.
    - Number of signals filtered by each new risk filter.
    - Score distribution of generated signals.

---

## 4. Key Performance Indicators (KPIs)

This section defines the KPIs to measure the success of the new features.

### 4.1. False Breakout Reduction
- **[ ] KPI:** Percentage of Breakout Signals Filtered by No-Flip Policy.
  - **Definition:** `(Number of breakout signals blocked by the 'no-flip' filter / Total number of potential breakout signals) * 100`.
  - **Target:** Establish a baseline and monitor for a statistically significant percentage, indicating the filter is active.
  - **Measurement:** Log each time the filter is triggered and divide by the total number of breakout signals considered.

### 4.2. Performance Stability
- **[ ] KPI:** CPU and Memory Usage Profile.
  - **Definition:** Average and maximum CPU and Memory usage over a 24-hour period.
  - **Target:** No more than a 10% increase in baseline CPU/Memory usage compared to the previous version.
  - **Measurement:** Monitor the application process using system monitoring tools (e.g., `htop`, `Prometheus`) during the soak test and regular operation.