# S7-T9: Metrics & Telemetry Plan

This document outlines the key metrics and structured logging strategy for monitoring the health and performance of the strategy ensemble and portfolio risk management systems.

## 1. Key Metrics

The following metrics will be emitted to provide visibility into system behavior. They are categorized into counters and gauges.

### 1.1. Counters

Counters are monotonically increasing values that track the occurrence of specific events.

| Metric Name                           | Description                                                                 |
| ------------------------------------- | --------------------------------------------------------------------------- |
| `veto.cluster_limit_breached`         | Incremented each time a trade is vetoed due to cluster exposure limits.     |
| `veto.daily_loss_hard_stop`           | Incremented each time a trade is vetoed due to the daily loss hard stop.        |
| `veto.max_open_trades_breached`       | Incremented each time a trade is vetoed due to max open trades limit.       |
| `veto.insufficient_capital`           | Incremented each time a trade is vetoed due to insufficient capital.        |
| `trade.resized.up`                    | Incremented each time a trade size is increased.                            |
| `trade.resized.down`                  | Incremented each time a trade size is decreased.                            |
| `trade.placed`                        | Incremented for every new trade placed successfully.                        |
| `trade.closed`                        | Incremented for every trade that is closed.                                 |

### 1.2. Gauges

Gauges represent a value that can go up or down.

| Metric Name                           | Description                                                                 |
| ------------------------------------- | --------------------------------------------------------------------------- |
| `exposure.usd.cluster.{cluster_name}` | Current total USD exposure for a given asset cluster (e.g., `btc-eth-sol`).  |
| `exposure.usd.total`                  | Current total USD exposure across all positions.                            |
| `open_trades.current`                 | The current number of open trades.                                          |
| `hit_rate.strategy.{strategy_name}`   | The hit rate (ratio of winning trades) for a specific strategy.             |
| `ensemble.agreement_rate`             | The percentage of strategies that agree on a given trade signal.            |

## 2. Structured Log Lines

Logs will be emitted in a structured, machine-parseable format (e.g., JSON) to facilitate automated analysis and alerting.

### 2.1. Trade Veto Event

This log is generated when a trade is vetoed by a risk filter.

```json
{
  "timestamp": "2023-10-27T10:00:00Z",
  "level": "WARNING",
  "component": "RiskManager",
  "event_type": "TradeVetoed",
  "details": {
    "signal_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "strategy": "trend_pullback",
    "symbol": "BTC/USDT",
    "side": "long",
    "reason_code": "CLUSTER_LIMIT_BREACHED",
    "message": "Vetoed trade for BTC/USDT due to BTC-ETH-SOL cluster exposure limit.",
    "context": {
      "cluster": "btc-eth-sol",
      "current_exposure_usd": 150000,
      "limit_usd": 100000
    }
  }
}
```

### 2.2. Trade Resize Event

This log is generated when a trade's size is adjusted by the sizing module.

```json
{
  "timestamp": "2023-10-27T10:05:00Z",
  "level": "INFO",
  "component": "PositionSizer",
  "event_type": "TradeResized",
  "details": {
    "signal_id": "b2c3d4e5-f6a7-8901-2345-67890abcdef1",
    "strategy": "mean_reversion_v2",
    "symbol": "ETH/USDT",
    "side": "short",
    "reason_code": "VOLATILITY_ADJUSTMENT",
    "message": "Resized trade for ETH/USDT based on updated volatility.",
    "context": {
      "initial_size_usd": 50000,
      "final_size_usd": 45000,
      "volatility_metric": 0.045
    }
  }
}
```

### 2.3. Ensemble Agreement Event

This log is generated to record the outcome of the ensemble voting process for a given signal.

```json
{
  "timestamp": "2023-10-27T10:10:00Z",
  "level": "INFO",
  "component": "EnsembleEngine",
  "event_type": "EnsembleVoteConducted",
  "details": {
    "signal_id": "c3d4e5f6-a7b8-9012-3456-7890abcdef12",
    "symbol": "SOL/USDT",
    "side": "long",
    "agreement_rate": 0.83,
    "total_votes": 6,
    "agreeing_votes": 5,
    "participating_strategies": ["trend_pullback", "mean_reversion_v2", "funding_premium", "liquid_momentum", "ob_absorption_v1", "regime_filter"],
    "outcome": "signal_approved"
  }
}