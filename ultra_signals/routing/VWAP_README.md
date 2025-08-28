VWAP Executor (overview)
=========================

This document summarizes the runtime knobs exposed by the local `VWAPExecutor`
implementation and the telemetry / FeatureView fields it emits per slice.

Configuration knobs
- volume_curve: list[float] - percent-of-day fractions per bin. Must sum to ~1.0. Default: internal U-shape.
- pr_cap: float - participation rate cap (fraction of ADV) used to bound slice notional. Default: 0.1 (10%).
- jitter_frac: float - relative jitter for slice size/time (e.g. 0.05 -> ±5%). Default: 0.05.
- max_slice_notional: Optional[float] - absolute cap per slice (USD or notional). Default: None.
- rtt_map: dict[venue->ms] - RTT map used by the StrategySelector cost estimates.
- feature_provider: callable(symbol)->dict - optional function returning live feature dict (keys below) used by Style Switcher.

Style switch thresholds (in config/vwap section)
- style_hysteresis_margin_bps: float - minimum improvement margin to switch styles (default 0.2 bps).

Telemetry / FeatureView fields (per-slice)
The executor emits per-slice telemetry events via the routing `TelemetryLogger` and can write
compact FeatureView records using the `FeatureViewWriter.write_record` API.

Common fields emitted:
- ts: timestamp (ms)
- symbol: instrument
- slice_index / bin: slice ordinal
- requested_slice_notional: target notional derived from volume curve
- slice_notional: actual notional executed (after PR cap & jitter)
- exec_strategy: one of {MARKET, LIMIT, TWAP, VWAP}
- router_allocation: dict venue->pct selected for the slice
- expected_cost_bps: estimated all-in cost (bps) for the chosen allocation
- realized_cost_bps: (if writer provided) the fill price-derived cost in bps
- schedule_lag_lead: signed notional/time delta vs schedule (optional)
- components: JSON blob for extended debugging (vpin, lambda, spread_z, reason codes)

Notes
- The FeatureViewWriter schema is forward-compatible: extra keys will be stored in `components` and DB columns
  will be added when first written.
- Keep style-switch hysteresis configured reasonably to avoid frequent flip-flop.
