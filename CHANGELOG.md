## Unreleased

- Add online EWKyleEstimator and TimeWindowAggregator for per-symbol lambda estimation.
- FeatureStore: optional signed notional regression (features.impact.use_notional) and new flags:
  - features.impact.use_trade_price
  - features.impact.invert_notional_sign
  - features.impact.history_window
- BrokerSim: accepts lambda provider and applies temporary impact when available.
## Unreleased - Sprint 50

- Add online Kyle lambda estimator (EWKyleEstimator) with confidence intervals and lambda_z significance metric.
- FeatureStore: emit `impact` feature group per symbol with lambda estimates and adapter-derived hints.
- ImpactAdapter: map lambda_z to execution hints (participation %, passive preference, size multipliers).
- BrokerSim: accept lambda provider callable and apply temporary impact ΔP_temp = k_temp * λ * slice_volume when available.
- Add integration test to assert BrokerSim slippage correlates with lambda bins.
