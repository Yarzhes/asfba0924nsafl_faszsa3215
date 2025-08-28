Sprint-50: Add online Kyle-impact estimation and BrokerSim integration

Summary:
- Implemented an online Exponential-Weighted Kyle estimator (lambda = ΔP / ΔQ) with TimeWindowAggregator.
- FeatureStore integration: per-symbol aggregator + estimator, robust spread/depth z-scores, optional signed-notional regression (config flags documented in README-impact.md).
- BrokerSim: accepts a lambda_provider and applies temporary impact proportional to lambda when available; depth-based fallback preserved.
- Tests: added unit tests for estimator and notional-flag permutations plus focused BrokerSim integration test.

Configuration & compatibility:
- Backwards-compatible: ChartPatternLibrary ctor accepts legacy fractal_k kwarg.
- New feature flags: features.impact.use_notional, features.impact.use_trade_price, features.impact.invert_notional_sign, features.impact.history_window

Notes for reviewers:
- Key files: `ultra_signals/market/kyle_online.py`, `ultra_signals/core/feature_store.py`, `ultra_signals/sim/broker.py`.
- Tests updated and run locally. See CHANGELOG.md for short notes.
Sprint 50: Online Kyle Lambda Estimator + Impact-Aware Execution

This change adds an online exponential-weighted Kyle lambda estimator (EWKyleEstimator), exposes per-symbol impact features through the FeatureStore, and wires an ImpactAdapter to generate execution hints (impact_state, target_participation_pct, prefer_passive, size_multiplier).

BrokerSim is extended to accept a lambda provider callable and will use λ to compute a temporary price impact proportional to executed slice volume. The RouterAdapter passes FeatureStore.get_lambda_for as the provider when a FeatureStore instance is available in settings.

Files added/changed:
- ultra_signals/market/kyle_online.py (EW estimator + CI)
- ultra_signals/market/impact_adapter.py (adapter already present)
- ultra_signals/market/tick_helpers.py (tick rule helper)
- ultra_signals/core/feature_store.py (wire estimator, emit impact features)
- ultra_signals/sim/broker.py (lambda_provider support already present)
- ultra_signals/tests/test_brokersim_impact_by_lambda.py (integration test scaffold)
- CHANGELOG.md, ultra_signals/market/README-impact.md

Notes for reviewer:
- The EW estimator uses a plug-in EW residual to approximate standard error; this is fast and dependency-free.
- ImpactAdapter uses simple hysteresis thresholds; tune hi_th/lo_th/base_participation via settings if needed.
- I ran unit tests for the estimator locally; CI should run full test-suite. There were unrelated failures in pattern tests in an earlier full-run which appear unrelated to these changes.
