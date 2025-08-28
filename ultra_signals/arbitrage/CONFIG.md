# Arbitrage Module Config (Sprint 47)

Example settings section:

```yaml
arbitrage:
  enabled: true
  poll_interval_sec: 2.0
  notional_buckets_usd: [5000, 25000, 50000]
  min_after_cost_bps: 1.5          # minimum executable spread (after cost model) to flag
  basis_threshold_bps: 5.0         # absolute basis bps to flag
  geo_premium_threshold_bps: 2.0   # absolute geo premium bps to flag
  geo_baskets:
    US: [coinbase, kraken]
    ASIA: [binance_usdm, bybit_perp, okx_swap]
  venue_regions:
    binance_usdm: ASIA
    bybit_perp: ASIA
    okx_swap: ASIA
    coinbase_spot: US
    kraken_spot: US
  fee_overrides_bps:               # optional static maker/taker fee assumptions (bps)
    binance_usdm: { maker: 1.0, taker: 5.0 }
    bybit_perp: { maker: 1.0, taker: 5.0 }
  risk:
    latency_penalty_bps: 0.5
    maintenance_veto: true
```

Notes:
- `venue_regions` used for geo premium; falls back to basket membership.
- Depth walk for executable spread currently approximated by top-of-book.
- Funding snapshots pending integration with existing `FundingProvider` or public endpoints.
- Risk scoring is a minimal heuristic; extend with volatility + persistence signals.
