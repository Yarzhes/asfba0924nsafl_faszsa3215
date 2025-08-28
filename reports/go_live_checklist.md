# Go-Live Checklist - Sniper Mode v0.9.2

## Environment Configuration

### Core Settings
- **Version**: v0.9.2-sniper-safe
- **Mode**: Production Ready
- **Commit Hash**: `dda909b`
- **Tag**: `v0.9.2-sniper-safe`

### Sniper Mode Configuration
```yaml
sniper_mode:
  enabled: true
  mtf_confirm: true
  max_signals_per_hour: 2
  daily_signal_cap: 6
  cooldown_bars: 10
```

### Redis Configuration
```yaml
redis:
  enabled: true
  host: "localhost"
  port: 6379
  db: 0
  password: null  # Set via REDIS_PASSWORD env var
  timeout: 5.0
```

### Prometheus Metrics
```yaml
prometheus:
  enabled: true
  port: 8000
  path: "/metrics"
```

## Symbol & Venue Configuration

### Target Symbols (20 pairs)
- BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT
- DOGEUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, TONUSDT  
- TRXUSDT, DOTUSDT, NEARUSDT, ATOMUSDT, LTCUSDT
- BCHUSDT, ARBUSDT, APTUSDT, MATICUSDT, SUIUSDT

### Timeframes
- Primary: 5m
- Multi-timeframe: 1m, 3m, 5m, 15m
- MTF confirmation required for signals

### Venue Allowlist
- Primary: Binance USD-M Futures
- Backup: (to be configured in production)

## Risk Controls

### Position Sizing Caps
- Max risk per trade: 0.010% (1% of equity)
- Max signals per hour: 2
- Max signals per day: 6
- Cooldown between signals: 10 bars

### Vetoes Active
- VPIN high liquidity events
- Lambda (Kyle) extreme readings  
- Funding rate spike protection
- Circuit breaker integration
- MTF confluence requirement

### Safety Limits
- Max leverage: 10x
- Max spread tolerance: 0.06%
- Min confidence threshold: 0.60
- Auto-pause on circuit breaker

## Rollback Plan

### Emergency Revert
```bash
git checkout v0.9.2-sniper-safe
# Restart with safe settings
```

### Gradual Rollback
1. Disable sniper_mode.enabled in settings.yaml
2. Increase caps gradually if needed  
3. Monitor for 30 minutes before next change

### Circuit Breaker Triggers
- Daily loss > 4%
- Consecutive losses > 4
- Latency p95 > 2000ms
- Redis connection failures

## Monitoring & Alerts

### Grafana Dashboard
- Import: `dashboards/sniper-mode-dashboard.json`
- Key panels: rejection rates, latency, signal counts

### Prometheus Metrics
- `sniper_rejections_total{reason}`
- `signals_candidates_total`  
- `signals_blocked_total{reason}`
- `latency_tick_to_decision_ms`

### Telegram Notifications
- Format: "LIVE | {pair} | {side} | ENTRY:{entry} | SL:{sl} | TP:{tp} | sniper:{status}"
- Pre-trade signals only for allowed trades
- No trade cards for blocked signals

## Deployment Steps

### 1. Pre-deployment
- [ ] Verify Redis server running
- [ ] Test Prometheus endpoint
- [ ] Confirm Telegram bot configured
- [ ] Check API credentials in env

### 2. Shadow Mode (90-120 min)
- [ ] Run `python scripts/run_shadow_test.py`
- [ ] Monitor with `python scripts/monitor_shadow.py`
- [ ] Verify sniper caps enforce correctly
- [ ] Check MTF confirmation blocks signals
- [ ] Confirm no crashes/memory leaks

### 3. Canary Mode (1 symbol, 1 hour)
- [ ] Enable live orders for BTCUSDT only
- [ ] Set strict caps: 1 entry/hour, 2/day
- [ ] Monitor fills, slippage, rejection metrics
- [ ] Verify Telegram trade cards correct
- [ ] Check auto-pause on circuit breakers

### 4. Full Production
- [ ] Enable all 20 symbol pairs
- [ ] Normal sniper caps: 2/hour, 6/day
- [ ] Full venue allowlist active
- [ ] 24/7 monitoring dashboard

## Success Criteria

### Shadow Mode
- ✅ Sniper caps enforced (visible in metrics)
- ✅ MTF disagreements block signals  
- ✅ No crashes for 90+ minutes
- ✅ Latency p95 < 500ms
- ✅ Prometheus metrics updating

### Canary Mode  
- ✅ Fills within expected slippage tolerance
- ✅ Trade cards show correct data
- ✅ Sniper rejections tracked in metrics
- ✅ Circuit breakers respond appropriately
- ✅ Redis counters persist across restarts

### Production
- ✅ All 20 pairs operational
- ✅ Feature propagation: Collectors → Feature Store → Vetoes → Router
- ✅ TCA reports generated successfully
- ✅ No regression in existing functionality

## Emergency Contacts

- **Primary**: Trading Operations Team
- **Secondary**: Infrastructure Team  
- **Escalation**: Engineering Leadership

---
*Generated: August 28, 2025*
*Version: v0.9.2-sniper-safe*
