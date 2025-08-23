# S4-T7: Funding Window Filter Policy

**Owner:** Architect
**Status:** Defined

## 1. Overview

This document specifies the policy for the funding window risk filter. The purpose of this filter is to prevent taking new trades on a symbol during a configurable time window around its scheduled funding event. This is a risk management measure to avoid exposure to potential volatility and unpredictable price movements associated with funding rate payments.

## 2. Configuration Parameter

The filter's behavior is controlled by a single configuration parameter.

*   **Parameter:** `avoid_funding_minutes`
*   **Description:** The number of minutes before and after the next funding timestamp during which new entries are disallowed.
*   **Default Value:** `5`
*   **Configuration File:** This parameter must be configurable in `settings.yaml` under the risk management or strategy section.

## 3. "No-New-Entries" Rule

### 3.1. Rule Definition

**Rule:** No new entry positions will be taken for a symbol if the current time is within the window defined by `now - avoid_funding_minutes <= funding_timestamp <= now + avoid_funding_minutes`.

Where:
*   `now` is the current timestamp when the entry signal is evaluated.
*   `funding_timestamp` is the next scheduled funding event for the symbol.
*   `avoid_funding_minutes` is the configured avoidance period.

### 3.2. Scope and Exceptions

*   **In Scope:** This rule applies exclusively to signals that would open a new position (i.e., new entries).
*   **Out of Scope (Exceptions):** This rule **does not** apply to signals that close, reduce, or manage an existing position. Position-closing logic will operate independently of this filter.

## 4. Policy Examples & Test Cases

The following examples illustrate the filter's behavior, assuming `avoid_funding_minutes` is set to its default value of `5`.

**Scenario Configuration:**
*   `avoid_funding_minutes`: 5
*   Next Funding Time for `BTC/USDT:USDT`: `12:00:00 UTC`

| Case | Current Time (UTC) | Signal Type | Symbol | Is Entry Allowed? | Justification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1. Outside Window (Before) | `11:54:59 UTC` | Entry |`BTC/USDT:USDT`| Yes | More than 5 minutes before the funding event. |
| 2. Inside Window (Before) | `11:55:01 UTC` | Entry |`BTC/USDT:USDT`| No  | Within the 5-minute window before the funding event. |
| 3. Exit Signal (Inside Window) | `11:56:00 UTC` | Exit |`BTC/USDT:USDT`| N/A |Allowed; the rule does not apply to exit signals. |
| 4. Boundary Case (Start) | `11:55:00 UTC` | Entry |`BTC/USDT:USDT`| No  | Exactly at the start of the avoidance window boundary. |
| 5. Outside Window (After) | `12:05:01 UTC` | Entry |`BTC/USDT:USDT`| Yes | More than 5 minutes after the funding event. |
| 6. Inside Window (After) | `12:04:59 UTC` | Entry |`BTC/USDT:USDT`| No  | Within the 5-minute window after the funding event. |
| 7. Boundary Case (End) | `12:05:00 UTC` | Entry |`BTC/USDT:USDT`| No  | Exactly at the end of the avoidance window boundary. |
