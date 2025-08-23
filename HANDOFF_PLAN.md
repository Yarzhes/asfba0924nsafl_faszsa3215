# Sprint 7 Handoff Plan

This document provides a clear handoff to the implementation team, outlining the design and the plan for building the features for Sprint 7.

## 1. Overview

The core of this sprint is to introduce an ensemble-based decision-making process, backed by a robust portfolio management and risk control layer. This will allow the system to synthesize signals from multiple strategies, manage risk more holistically, and operate within predefined safety constraints.

## 2. Design Documents

The complete design is detailed in the following documents:

*   **`S7_DESIGN_BRIEF.md`**: The original requirements document.
*   **`CONFIG_SCHEMA.md`**: Defines the new configuration sections (`ensemble`, `correlation`, `portfolio`, `brakes`, `vol_risk_scale`). All validation and implementation should conform to this schema.
*   **`DATA_MODELS.md`**: Details the structure of the core data objects: `SubSignal`, `EnsembleDecision`, `PortfolioState`, and `RiskEvent`. These are the primary data containers that will flow through the new system.
*   **`API_INTERFACES.md`**: Specifies the function signatures for all new and updated modules. This is the contract for how the different components will interact.
*   **`TEST_PLAN.md`**: Outlines the full suite of tests that need to be implemented to ensure the quality and correctness of the new features.

## 3. Implementation Flow

The recommended implementation sequence is as follows:

1.  **Build all `SubSignals` per strategy:** Each strategy file in `strategies/` should be updated to implement `generate_subsignal`.
2.  **Develop the `Ensemble` engine:** Create `engine/ensemble.py` to merge `SubSignals` into an `EnsembleDecision`.
3.  **Implement `Portfolio` evaluation:** Create `risk/portfolio.py` to handle exposure checks and other portfolio-level constraints.
4.  **Integrate Portfolio and Brakes:** Update `engine/risk_filters.py` to incorporate the new portfolio context and risk brakes.
5.  **Apply Volatility Scaling:** Update `engine/sizing.py` with the `apply_volatility_scaling` function.
6.  **Integrate with Backtester:** Update `backtest/event_runner.py` to use the new portfolio evaluation logic to gate and resize trades.
7.  **Update Telegram Transport:** Modify `transport/telegram.py` to include the enhanced notifications.
8.  **Implement all tests:** Concurrently with the steps above, implement the tests outlined in `TEST_PLAN.md`.

## 4. Key Considerations

*   **Performance:** Pay close attention to the performance budgets specified in the brief. The correlation refresh should be handled asynchronously, and the ensemble/portfolio checks must be highly efficient.
*   **Fallback Path:** Ensure that if `ensemble.enabled` is `false`, the system gracefully falls back to the previous single-engine execution path.
*   **Logging:** All `RiskEvent`s and `EnsembleDecision` `vote_detail` should be logged clearly for later analysis.

This set of documents provides a complete guide for the development work. Please review them carefully before starting implementation.