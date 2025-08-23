# Performance Budgets & Reliability Requirements

This document outlines the key performance indicators (KPIs), latency budgets, and reliability standards for the new derivatives and regime engine features.

## 1. Performance Budgets

### 1.1. Data Provider Latency

*   **OI Provider Refresh:** The call to `OIProvider.fetch_oi_delta` must complete within the configured `derivatives.oi.refresh_sec`. The default is **60 seconds**.
*   **Compute Path Latency:** The synchronous feature computation path that runs on every tick (e.g., merging OI data, calculating the `derivatives_score`) **must not block the main WebSocket event loop**.
    *   **Budget:** The entire per-tick computation slice for these new features must be well under **10 ms**.
    *   **Mitigation:** `fetch_oi_delta` is an `async` function and must be called in a separate task, not directly in the hot path. The results are then merged into the feature vector when they become available.

### 1.2. Memory Usage

*   **Force Order Aggregation:** The `compute_liq_pulse` function will process a stream of liquidation events. The underlying data structure holding these events (e.g., a `collections.deque`) must have a bounded size to prevent memory leaks.
    *   **Budget:** The total memory footprint for storing force order events for **20 symbols** should not exceed **25 MB**.
    *   **Mitigation:** The deque should be sized based on the `liq_pulse.window_sec` to only store relevant recent events.

## 2. Reliability & Fault Tolerance

### 2.1. OI Provider Failure

*   **Requirement:** The signal engine must remain fully operational even if the configured OI provider (`coinglass`, `coinalyze`, etc.) is down or returns an error.
*   **Degradation Strategy:**
    1.  If `fetch_oi_delta` fails (e.g., timeout, API error), the function must catch the exception and log it appropriately.
    2.  It should then return a neutral or "zero" value for all OI-related features (e.g., `oi_delta_1m = 0`, `oi_delta_5m = 0`).
    3.  The `derivatives_score` will consequently have a neutral `oi_score`, effectively removing it from the final calculation without crashing the engine.
    4.  The system should periodically attempt to reconnect to the provider.

### 2.2. Funding Data Staleness

*   **Requirement:** The system should be robust to stale funding rate data.
*   **Degradation Strategy:** If the funding rate trail fails to refresh, the engine will continue to use the last known values. The `funding_now` component will be disabled if the mark price stream is also unavailable, leading to a neutral `funding_score`.

## 3. Monitoring & Observability

*   **Metrics:** The following metrics should be exposed to monitor the health of these new components:
    *   `oi_provider_requests_total{provider, symbol, status}` (e.g., status="success" or "failure")
    *   `oi_provider_latency_seconds{provider, symbol}` (histogram)
    *   `liq_events_processed_total{symbol}`
    *   `regime_changes_total{profile}` (counter for how often the regime profile changes)
    *   `signals_blocked_total{reason}` (e.g., reason="funding_window")