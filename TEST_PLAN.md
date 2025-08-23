# Test Plan

This document outlines the testing strategy for the features developed in this sprint.

## Unit Tests

### `test_subsignals_per_strategy.py`
- For each strategy, verify that `generate_subsignal` produces a `SubSignal` with the correct structure.
- Test that `confidence_calibrated` is within the [0, 1] range.
- Ensure the `reasons` dictionary contains plausible data for both LONG and SHORT signals.

### `test_ensemble_weighted_vote_and_veto.py`
- Test the weighted voting mechanism with various `SubSignal` inputs and `regime_profile` settings.
- Verify that `EnsembleDecision` is correctly formed and that `confidence` reflects the weighted sum.
- Test each veto condition (`breakout_requires_book_flip`, `mr_requires_band_pierce`) to ensure they correctly prevent a decision.

### `test_correlation_groups_hysteresis.py`
- Test `compute_corr_groups` with a sample returns matrix to ensure it groups symbols correctly based on the `threshold`.
- Test `update_corr_state` to verify that group assignments only change after `hysteresis_hits` is met.

### `test_portfolio_exposure_caps_and_netting.py`
- Test `evaluate_portfolio` against various `PortfolioState` scenarios.
- Verify that `max_risk_per_symbol`, `max_positions_per_symbol`, `max_positions_total`, `max_cluster_risk`, `max_net_long_risk`, `max_net_short_risk` and `margin_cap_pct` are all respected.
- Ensure that a `(False, 0.0, ...)` tuple is returned for breaches, along with the correct `RiskEvent`.

### `test_brakes_spacing_daily_loss_streaks.py`
- Test the `min_spacing_sec_same_cluster` brake.
- Test the `daily_loss_soft_pct` and `daily_loss_hard_pct` brakes.
- Test the cooldown logic for `cooldown_after_streak_symbol` and `cooldown_after_streak_global`.
- Ensure appropriate `RiskEvent` objects are generated for each brake condition.

### `test_volatility_scaled_sizing.py`
- Test `apply_volatility_scaling` with different `atr_percentile` values.
- Verify that `low_vol_boost` is applied when volatility is below `low_vol_pct`.
- Verify that `high_vol_cut` is applied when volatility is above `high_vol_pct`.

### `test_telegram_vote_summary.py`
- Mock the Telegram transport and verify that the message content includes the ensemble vote summary and veto reasons when applicable.

## Integration Tests

### `test_backtest_portfolio_integration.py`
- Run a full backtest with the ensemble and portfolio modules enabled.
- Verify that trades are correctly gated or downsized based on portfolio evaluation.
- Check the backtest logs to ensure `RiskEvent` objects are being recorded correctly.

## Acceptance Checklist
- [ ] Config validates with defaults; turning `ensemble.enabled=false` falls back to single-engine path.
- [ ] Ensemble outputs 1 decision with clear reasons; vetoes logged.
- [ ] Portfolio caps & brakes enforce as configured in both **live** and **backtest**.
- [ ] Walk-forward runs with ensemble + portfolio, producing KPIs per strategy and overall.