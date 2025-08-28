Smart Multi-Venue Router (minimal skeleton)

This folder contains a small, testable starting point for Sprint 55 features:

- Aggregator: holds per-venue L2 snapshots and depth walk (VWAP)
- Cost model: simple fees + impact + latency penalty
- Router: selects or splits across venues; integrates with HealthMonitor and supports circuit-breaking
- TWAPExecutor: uses Router per slice to allocate child notional
- HealthMonitor: heartbeat-based health checks

Example
-------
Run the example TWAP script:

```bash
python examples/example_twap.py
```

Tests
-----
Run unit tests (no pytest required):

```bash
python -m tests.run_tests
```

Notes
-----
This is intentionally minimal and contains toy models. Replace cost/impact logic with production models and wire telemetry and order submission in the execution layer.
