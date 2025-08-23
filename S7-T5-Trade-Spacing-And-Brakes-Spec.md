# S7-T5: Trade Spacing and Performance Brakes Specification

**Author:** Roo, Technical Leader
**Date:** 2025-08-22
**Status:** Final

## 1. Overview

This document specifies the policies for dynamic, event-driven risk controls designed to complement the static exposure caps defined in `S7-T4`. The two primary controls are:

1.  **Trade Spacing:** Prevents over-concentration in a single asset cluster by enforcing a time delay between new trade entries.
2.  **Performance Brakes:** Implements automated controls to reduce or halt trading activity in response to losses, both at a daily level and in response to consecutive losing streaks.

These rules are designed to be evaluated by the `RiskEngine` before any new trade is submitted to the market.

## 2. Trade Spacing Rules

To prevent "dogpiling" into a single theme or asset class too rapidly, a minimum time-based spacing is enforced between new trade entries within the same **asset cluster**.

*   **Definition of Asset Cluster:** An asset cluster is a group of highly correlated instruments (e.g., `[BTC, ETH, SOL]` or `[MEME_COIN_A, MEME_COIN_B]`). The cluster definitions are maintained outside this policy.
*   **Rule:** The system will block a new trade entry if another trade was initiated in the same asset cluster within the last **3 minutes**.

## 3. Performance Brakes

Performance brakes are applied based on daily performance and consecutive losing streaks. These are designed to protect capital and prevent emotion-driven trading after a series of losses.

### 3.1. Daily Loss Limits

Two levels of daily loss limits are applied based on the Net P&L for the current trading session (UTC day).

*   **Soft Stop:** If the daily Net P&L breaches **-2%** of the portfolio's start-of-day equity, a risk reduction is triggered.
*   **Hard Stop:** If the daily Net P&L breaches **-4%** of the portfolio's start-of-day equity, all new trading activity is halted for the remainder of the session.

### 3.2. Losing Streak Cooldowns

Cooldowns are triggered by consecutive losing trades to provide a "time out" for specific symbols or the entire portfolio. A trade is considered a loss for streak-counting purposes if it closes with a negative Realized P&L.

*   **Symbol-Specific Streak:** A cooldown is applied to a specific symbol after a set number of consecutive losses.
*   **Global Streak:** A portfolio-wide cooldown is applied after a longer streak of consecutive losses across any symbol.

## 4. Truth Table: Triggers and Actions

The following table summarizes the risk control triggers and their corresponding preventative actions. The `RiskEngine` will evaluate these rules in order, and the first rule that matches will trigger its corresponding action.

| # | Trigger Event | Condition | Action | Scope | Cooldown/Duration |
|---|---|---|---|---|---|
| 1 | **Daily Hard Stop** | `Daily Net P&L <= -4%` | **Halt All New Trades** | Global | Remainder of UTC Day |
| 2 | **Daily Soft Stop** | `Daily Net P&L <= -2%` | **Reduce Max Risk per Trade by 50%** | Global | Until EOD or P&L Recovers |
| 3 | **Global Losing Streak** | `5 consecutive losses` | **Halt All New Trades** | Global | 4 Hours |
| 4 | **Symbol Losing Streak** | `3 consecutive losses on Symbol X` | **Halt New Trades for Symbol X** | Symbol | 1 Hour |
| 5 | **Trade Spacing** | `New trade in Cluster Y within 3 mins of last` | **Block New Trade** | Asset Cluster | 3 Minutes |

---