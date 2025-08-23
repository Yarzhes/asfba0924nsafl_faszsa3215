# API Interfaces

This document defines the public-facing functions (API) for each new or updated module.

## `strategies/*.py`

Each strategy is implemented in its own file and must expose the following function:

### `generate_subsignal`

Generates a `SubSignal` based on the provided feature vector and context.

- **Signature:** `def generate_subsignal(fv, ctx) -> SubSignal`
- **Parameters:**
    - `fv` (object): A feature vector or data container with the necessary indicators for the strategy.
    - `ctx` (object): A context object containing additional information like symbol, timeframe, etc.
- **Returns:** A `SubSignal` object.

---

## `engine/ensemble.py`

This new module is responsible for combining signals from multiple strategies.

### `combine_subsignals`

Merges a list of `SubSignal` objects into a single `EnsembleDecision`.

- **Signature:** `def combine_subsignals(subs: list[SubSignal], regime_profile: str, settings) -> EnsembleDecision`
- **Parameters:**
    - `subs` (list[SubSignal]): A list of sub-signals from various strategies for the same symbol and timeframe.
    - `regime_profile` (str): The current market regime (e.g., "trend", "mr") used to select the appropriate weights.
    - `settings` (object): The `ensemble` section of the application configuration.
- **Returns:** An `EnsembleDecision` object representing the final verdict.

---

## `analytics/correlation.py`

This new module handles the calculation of symbol correlations to manage portfolio risk.

### `compute_corr_groups`

Calculates correlation-based clusters from a DataFrame of asset returns.

- **Signature:** `def compute_corr_groups(returns_df, threshold: float) -> dict[str, str]`
- **Parameters:**
    - `returns_df` (pd.DataFrame): A DataFrame where columns are symbols and rows are returns over time.
    - `threshold` (float): The correlation value above which symbols are considered part of the same group.
- **Returns:** A dictionary mapping each symbol to its cluster identifier (e.g., `{'BTC/USDT': 'cluster_1', 'ETH/USDT': 'cluster_1'}`).

### `update_corr_state`

Applies hysteresis to the correlation group updates to prevent rapid switching.

- **Signature:** `def update_corr_state(prev_state, new_groups, hysteresis_hits: int) -> dict[str, str]`
- **Parameters:**
    - `prev_state` (dict): The last known mapping of symbols to clusters.
    - `new_groups` (dict): The newly computed grouping.
    - `hysteresis_hits` (int): The number of consecutive computations the new group must appear in before it is accepted.
- **Returns:** The updated and confirmed symbol-to-cluster mapping.

---

## `risk/portfolio.py`

This new module evaluates trading decisions against portfolio constraints.

### `evaluate_portfolio`

Checks if a proposed trade (`EnsembleDecision`) is permissible based on current portfolio state and risk settings.

- **Signature:** `def evaluate_portfolio(decision: EnsembleDecision, state: PortfolioState, settings) -> tuple[bool, float, list[RiskEvent]]`
- **Parameters:**
    - `decision` (EnsembleDecision): The trade being considered.
    - `state` (PortfolioState): The current state of the portfolio.
    - `settings` (object): The `portfolio` section of the application configuration.
- **Returns:** A tuple containing:
    - `allowed` (bool): `True` if the trade can proceed.
    - `size_scale` (float): A scaling factor (0.0 to 1.0) to be applied to the trade size. A value less than 1.0 indicates a downsize.
    - `events` (list[RiskEvent]): A list of any risk events that were triggered during evaluation.

---

## `engine/sizing.py` (Update)

This existing module will be updated to include volatility-based scaling.

### `apply_volatility_scaling`

Adjusts the base risk for a trade based on the market's volatility percentile.

- **Signature:** `def apply_volatility_scaling(base_risk: float, atr_percentile: float, cfg) -> float`
- **Parameters:**
    - `base_risk` (float): The initial risk amount for the trade.
    - `atr_percentile` (float): The current ATR percentile (0-100).
    - `cfg` (object): The `vol_risk_scale` section of the config.
- **Returns:** The adjusted risk amount.

---

## `engine/risk_filters.py` (Update)

This existing module will be updated to incorporate new portfolio-level risk checks.

- The functions within this module will be updated to accept a `portfolio_ctx` object.
- They will be responsible for generating and appending `RiskEvent` objects for violations related to trade spacing, daily loss limits, and losing streaks.

---

## `backtest/event_runner.py` (Update)

The backtesting engine needs to integrate the new portfolio evaluation logic.

- It will call `risk.portfolio.evaluate_portfolio` for each potential trade.
- The result will be used to gate or resize simulated trades.
- All generated `RiskEvent` objects will be logged as part of the backtest results.

---

## `transport/telegram.py` (Update)

The Telegram notification transport will be updated to provide more detailed alerts.

- Alerts for new trades will optionally include a summary of the ensemble vote (`vote_detail`).
- If a trade was vetoed, the primary veto reason from the `vetoes` list will be included in the message.