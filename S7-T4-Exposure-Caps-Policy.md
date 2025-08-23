# S7-T4: Exposure Caps & Netting Policy

This document outlines the portfolio-level risk controls, including exposure limits, netting rules, and leverage constraints.

## 1. Core Exposure Limits

The following table defines the core risk limits for the portfolio. These caps are designed to prevent over-concentration in any single asset or cluster and to manage overall market exposure.

| **Limit Type** | **Description** | **Value** | **Scope** |
| :--- | :--- | :--- | :--- |
| **Per-Symbol Risk** | Maximum allowable risk per individual symbol, measured as a percentage of total equity. | `≤ 1.0%` | Single Symbol |
| **Cluster Notional (Long)** | Maximum notional exposure for all long positions within a correlated cluster. | `≤ 15.0%` of Equity | Asset Cluster |
| **Cluster Notional (Short)** | Maximum notional exposure for all short positions within a correlated cluster. | `≤ 15.0%` of Equity | Asset Cluster |
| **Net Portfolio Exposure**| Maximum net directional exposure (Long - Short) as a percentage of total equity. | `≤ 50.0%` | Entire Portfolio |
| **Gross Portfolio Exposure**| Maximum gross exposure (Long + Short) as a percentage of total equity. | `≤ 150.0%` | Entire Portfolio |
| **Leverage / Margin** | Maximum allowable leverage for the entire account. | `≤ 2.0x` | Account Level |

## 2. Edge Case Management

This section addresses potential edge cases and defines the protocol for handling them.

*   **Simultaneous Signals on Correlated Assets:**
    *   **Problem:** Multiple, simultaneous entry signals for highly correlated assets (e.g., BTC and ETH) could breach the `Cluster Notional` limit if positions are opened without checks.
    *   **Solution:** Before executing a new trade, the system will pre-calculate the post-trade cluster exposure. If the new position would cause a breach, the trade size will be scaled down proportionally to fit within the cluster limit. If the existing cluster exposure is already at its limit, the new signal will be ignored.
