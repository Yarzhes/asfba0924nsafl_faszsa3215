# S4: Sprint 4 Test Plan

## 1. Test Categories

### 1.1. Unit Tests

- [ ] **Regime Classifier:**
    - [ ] Verify that the classifier correctly identifies each defined market regime based on mock input data.
    - [ ] Test edge cases for regime transition thresholds.
    - [ ] Ensure the classifier outputs the expected regime string for each input.
- [ ] **Funding Filter Policy:**
    - [ ] Test that the `is_funding_imminent()` function correctly returns `True` within the specified time window before a funding event.
    - [ ] Test that the function returns `False` outside the funding window.
    - [ ] Verify that signal generation is correctly suppressed (returns `False` or a "filter" status) during the funding window.
- [ ] **OI Provider Interface:**
    - [ ] Test that the primary OI provider (`Coinalyze`) is selected by default.
    - [ ] Test the fallback mechanism: when the primary provider fails (mocked failure), verify the system switches to the secondary provider (`Bybit`).

### 1.2. Integration Tests

- [ ] **Regime Classifier & Weight Selection:**
    - [ ] Trigger a regime change and verify that the `Engine` correctly loads the corresponding weight profile from `settings.yaml`.
    - [ ] Confirm that generated signals use the new weights post-regime-change.
- [ ] **Funding Filter & Signal Generation:**
    - [ ] Run the system with live or replayed data and confirm that no new entry signals are generated during the configured funding-rate quiet windows.
    - [ ] Verify that other signals (e.g., exit signals) are not affected by this filter.
- [ ] **Data Provider Failover:**
    - [ ] Simulate an outage of the primary OI data provider.
    - [ ] Confirm the system logs the failure and switches to the fallback provider.
    - [ ] Confirm that the system continues to operate and generate signals using data from the fallback provider.

### 1.3. Soak Tests

- [ ] **Objective:** Run the system continuously for 24-48 hours to identify memory leaks, performance degradation, or crashes related to new polling/streaming features (Funding, OI).
- [ ] **Procedure Outline:**
    1.  Deploy the latest build to a staging environment.
    2.  Start the `realtime_runner.py` application.
    3.  Monitor system resource usage (CPU, Memory) over the test duration.
    4.  Continuously log key metrics:
        - Heartbeat/timestamp every 1 minute.
        - OI and Funding data points as they are received.
        - Any errors or warnings, especially related to data provider APIs.
    5.  At the end of the test, analyze logs for any anomalies, unhandled exceptions, or signs of resource leakage.
    6.  Verify that the application is still running and responsive.

## 2. Key Performance Indicators (KPIs) & Gates

- [ ] **KPI 1: Funding Window Signal Reduction:**
    - **Metric:** Number of new entry signals generated during the 15-minute window leading up to a funding event.
    - **Success Gate:** The number must be **zero**. All entry signals must be filtered during this period.
- [ ] **KPI 2: Data Provider Outage Resilience:**
    - **Metric:** System uptime and error state during a simulated primary data provider outage.
    - **Success Gate:** The system must **not crash or enter an unrecoverable error state**. It must log the outage, switch to the fallback provider, and continue operations within 2 minutes of the simulated failure.

## 3. Acceptance Checklist

| Test Case | Description | Pass/Fail |
|-----------|-------------|-----------|
| **Unit: Regime Classifier** | Classifier correctly identifies all regimes with mock data. | [ ] Pass / [ ] Fail |
| **Unit: Funding Filter** | `is_funding_imminent()` logic is correct for all time windows. | [ ] Pass / [ ] Fail |
| **Unit: OI Failover** | System correctly attempts primary, then secondary provider on mock failure. | [ ] Pass / [ ] Fail |
| **Integration: Regime Weights** | A regime change correctly triggers loading of the new weight profile. | [ ] Pass / [ ] Fail |
| **Integration: Funding Filter Live** | No entry signals are generated during funding quiet windows. | [ ] Pass / [ ] Fail |
| **Integration: OI Failover Live** | System successfully fails over to the secondary provider on a simulated outage. | [ ] Pass / [ ] Fail |
| **Soak Test: Stability** | System runs for 24+ hours without crashes or memory leaks. | [ ] Pass / [ ] Fail |
| **KPI: Funding Gate** | **Zero** entry signals are generated during funding windows over a 24-hour period. | [ ] Pass / [ ] Fail |
| **KPI: Outage Gate** | System remains operational and switches providers within 2 minutes of a simulated outage. | [ ] Pass / [ ] Fail |