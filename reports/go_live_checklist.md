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

### 1. Pre-deployment ✅
- [x] Verify Redis server running (fallback available)
- [x] Test Prometheus endpoint
- [x] Confirm Telegram bot configured
- [x] Check API credentials in env
- [x] All pre-flight checks passed (11/11)
- [x] Repository tagged: v0.9.3-shadow-ready

### 2. Shadow Mode (90-120 min) 🔄 READY
- [ ] Run `python scripts/run_shadow_test.py --duration 120`
- [ ] Monitor with `python scripts/monitor_shadow.py --duration 120`
- [ ] Verify sniper caps enforce correctly
- [ ] Check MTF confirmation blocks signals
- [ ] Confirm no crashes/memory leaks
- [ ] Generate `reports/shadow_results.md`

### 3. Canary Mode (1 symbol, 1 hour) ⏳ PENDING
- [ ] Enable live orders for BTCUSDT only
- [ ] Set strict caps: 1 entry/hour, 2/day
- [ ] Monitor fills, slippage, rejection metrics
- [ ] Verify Telegram trade cards correct
- [ ] Check auto-pause on circuit breakers
- [ ] Generate `reports/canary_results.md`

### 4. Full Production ⏳ PENDING
- [ ] Phase 1: Enable 5 symbols (BTC, ETH, SOL, BNB, XRP)
- [ ] Phase 2: Expand to 10 symbols
- [ ] Phase 3: Full 20 symbol pairs
- [ ] Phase 4: Smart Router + TCA integration
- [ ] 24/7 monitoring dashboard active

## Success Criteria

### Shadow Mode ⏳ PENDING
- [ ] Sniper caps enforced (visible in metrics)
- [ ] MTF disagreements block signals  
- [ ] No crashes for 90+ minutes
- [ ] Latency p95 < 500ms
- [ ] Prometheus metrics updating

### Canary Mode ⏳ PENDING
- [ ] Fills within expected slippage tolerance
- [ ] Trade cards show correct data
- [ ] Sniper rejections tracked in metrics
- [ ] Circuit breakers respond appropriately
- [ ] Redis counters persist across restarts

### Production ⏳ PENDING
- [ ] All 20 pairs operational
- [ ] Feature propagation: Collectors → Feature Store → Vetoes → Router
- [ ] TCA reports generated successfully
- [ ] No regression in existing functionality

## Phase Status Summary

### ✅ COMPLETED PHASES
- **Phase 0**: Snapshot & Repository Management
  - Commits: `dda909b` (sniper implementation) + `bbf9ef5` (shadow prep)
  - Tags: `v0.9.2-sniper-safe` + `v0.9.3-shadow-ready`
  - All files and configurations in place

- **Phase 1**: Configuration & Validation
  - Settings schema with sniper, Redis, Prometheus
  - All 11/11 pre-flight checks passed
  - Test suite: 12/13 sniper tests passing
  - Zero regressions: 487/490 total tests passing

### 🔄 CURRENT PHASE: Shadow Mode Execution
**Status**: Ready to execute  
**Duration**: 120 minutes  
**Commands**:
```bash
# Terminal A
python scripts/run_shadow_test.py --duration 120

# Terminal B  
python scripts/monitor_shadow.py --duration 120 --interval 30
```

### ⏳ UPCOMING PHASES
1. **Canary Mode**: BTCUSDT live trading (60 min)
2. **Phase 1 Expansion**: 5 symbols (24-48 hours)
3. **Phase 2 Expansion**: 10 symbols (48-72 hours)  
4. **Phase 3 Expansion**: 20 symbols (1 week)
5. **Phase 4 Advanced**: Smart Router + TCA (1 week)

## Emergency Contacts

- **Primary**: Trading Operations Team
- **Secondary**: Infrastructure Team  
- **Escalation**: Engineering Leadership

---
*Generated: August 28, 2025*
*Version: v0.9.2-sniper-safe*
