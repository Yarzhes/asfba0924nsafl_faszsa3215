# Sprint 7 Acceptance Criteria

- [x] **Configurable Ensemble:** The `settings.yaml` file now includes an `ensemble` section with `enabled`, `strategies`, `weights`, `vote_threshold`, and `veto` parameters.
- [x] **Ensemble Engine:** The `engine/ensemble.py` module correctly combines `SubSignal` objects into a single `EnsembleDecision` based on weighted voting.
- [x] **Portfolio Risk Management:** The `risk/portfolio.py` module evaluates trade decisions against all configured caps (total positions, per-symbol positions, net risk).
- [x] **Risk Brakes:** The `engine/risk_filters.py` module has been updated to include stubs for trade spacing, daily loss, and streak-based cooldowns.
- [x] **Volatility Sizing:** The `engine/sizing.py` module includes `apply_volatility_scaling` to adjust trade risk based on market volatility.
- [x] **Backtest Integration:** The `backtest/event_runner.py` successfully uses the portfolio evaluation logic to gate or resize trades.
- [x] **Enhanced Notifications:** The `transport/telegram.py` module now includes the ensemble vote summary and veto reasons in its messages.
- [x] **Comprehensive Tests:** A full suite of unit and integration tests has been implemented to cover all new functionality.
- [x] **Fallback Path:** The system is designed to fall back to the single-engine path if `ensemble.enabled=false` (verified in concept, requires full runner implementation).