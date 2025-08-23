# S4-T1: Funding Integration Plan

This document outlines the design for integrating funding rate data into the system.

## 1. API Contract

This section defines the data structures for funding rate information. The data will be sourced from a REST endpoint and merged with the `markPrice` stream.

### 1.1 Funding Rate Snapshot

The `FundingRateSnapshot` provides a real-time view of the current funding rate, delivered alongside the mark price for a specific symbol.

**YAML Definition:**
```yaml
FundingRateSnapshot:
  type: object
  properties:
    symbol:
      type: string
      description: The market symbol (e.g., BTCUSDT).
    mark_price:
      type: number
      description: The current mark price.
    funding_rate:
      type: number
      description: The current funding rate.
    next_funding_time:
      type: integer
      format: int64
      description: The Unix timestamp (milliseconds) for the next funding event.
    timestamp:
      type: integer
      format: int64
      description: The Unix timestamp (milliseconds) when the snapshot was generated.
```

### 1.2 Funding Rate History

The `FundingRateHistory` provides a trailing view of the last 8 funding periods. This data is intended for historical analysis and feature calculation.

**YAML Definition:**
```yaml
FundingRateHistory:
  type: object
  properties:
    symbol:
      type: string
      description: The market symbol (e.g., BTCUSDT).
    history:
      type: array
      description: A list of the last 8 funding rate periods.
      items:
        type: object
        properties:
          funding_rate:
            type: number
            description: The funding rate for the period.
          funding_time:
            type: integer
            format: int64
            description: The Unix timestamp (milliseconds) of the funding event.
```

## 2. Caching & Refresh

This section describes the caching strategy and refresh cadence for the funding rate history.

### 2.1 Caching Mechanism

A local in-memory cache will be used to store the `FundingRateHistory` for each symbol. This avoids repeated calls to the REST endpoint and ensures that historical data is readily available for feature computation. The cache will be keyed by symbol.

### 2.2 Refresh Cadence

The funding rate history will be refreshed periodically. The refresh interval will be configurable in the [`settings.yaml`](settings.yaml:1) file.

*   **Configuration:**
    ```yaml
    funding_rate_provider:
      refresh_interval_minutes: 15
    ```
*   **Default:** The default refresh interval will be 15 minutes.
*   **Behavior:** A background task will be responsible for fetching the funding rate history from the provider's REST endpoint at the specified interval. Upon a successful fetch, the cache will be updated with the new data.

## 3. Error Handling

This section details the system's behavior for various error conditions related to the funding rate data provider.

### 3.1 Provider Outage

If the REST endpoint for funding history is unreachable (e.g., due to a network error, a 5xx status code, or a timeout), the system will:
*   Log the error with relevant details (e.g., symbol, timestamp, error message).
*   Continue to use the last successfully fetched data from the cache.
*   Retry the request at the next scheduled refresh interval.

### 3.2 Missing or Invalid Data

If the provider returns a successful response but the data is incomplete, invalid, or missing for a specific symbol:
*   **Empty History:** If the `history` array is empty, the system will log a warning and the cached data for that symbol will not be updated. The stale data will continue to be used until the next successful refresh.
*   **Incomplete History:** If the `history` array contains fewer than 8 periods, the system will use the partial data provided, log a warning, and pad the remaining history with `null` values.
*   **Invalid Data Types:** If any fields have incorrect data types (e.g., `funding_rate` is a string), the entire update for that symbol will be discarded. A critical error will be logged, and the stale data will be used.

## 4. Test Plan

This section outlines the plan to test the funding rate integration.

### 4.1 Unit Tests

*   **Cache Logic:**
    *   Verify that the cache is correctly updated after a successful data fetch.
    *   Test that the cache is not updated when a fetch fails.
*   **Data Parsing:**
    *   Test the successful parsing of a valid `FundingRateHistory` payload.
    *   Test the handling of malformed or incomplete JSON payloads.

### 4.2 Integration Tests

*   **Happy Path:**
    *   Simulate a successful fetch from the REST endpoint and verify that the `FundingRateHistory` cache is populated correctly.
*   **Provider Outage:**
    *   Use a mock server to simulate a provider outage (e.g., by returning a 503 status code).
    *   Verify that the system logs the error and continues to use stale cache data.
*   **Missing Data:**
    *   Simulate a provider response with an empty `history` array for a specific symbol.
    *   Verify that a warning is logged and the cache for that symbol is not updated.
*   **Incomplete Data:**
    *   Simulate a provider response with a `history` array containing fewer than 8 entries.
    *   Verify that the partial data is used and padded with `null`s.
*   **Invalid Data:**
    *   Simulate a provider response with invalid data types.
    *   Verify that a critical error is logged and the cache is not updated.
