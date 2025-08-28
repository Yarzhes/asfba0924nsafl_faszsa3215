# Rollout Plan - Sniper Mode Production Expansion

## Executive Summary
Gradual expansion plan from single-symbol canary to full 20-symbol production deployment with Smart Router and TCA feedback integration.

---

## 🎯 Rollout Phases

### Phase 1: Core 5 Symbols (BTC, ETH, SOL, BNB, XRP)
**Duration**: 24-48 hours  
**Caps**: 2/hour, 6/day  
**Risk**: Low

```yaml
runtime:
  symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
  sniper_mode:
    max_signals_per_hour: 2
    daily_signal_cap: 6
```

**Success Criteria**:
- [ ] All 5 symbols showing fresh data/features
- [ ] Sniper caps enforced across all symbols
- [ ] No venue-specific issues
- [ ] TCA metrics within tolerance
- [ ] Daily P&L within expected range

### Phase 2: Extended Portfolio (10 Symbols)
**Duration**: 48-72 hours  
**Caps**: 2/hour, 8/day  
**Risk**: Medium-Low

**Additional Symbols**: DOGEUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, TONUSDT

```yaml
runtime:
  symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", 
           "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "TONUSDT"]
  sniper_mode:
    max_signals_per_hour: 2
    daily_signal_cap: 8  # Slight increase for more symbols
```

**Success Criteria**:
- [ ] Correlation management working
- [ ] Portfolio risk within limits
- [ ] No resource exhaustion
- [ ] Feature store scaling properly

### Phase 3: Full Symbol Set (20 Symbols)
**Duration**: 1 week  
**Caps**: 3/hour, 10/day  
**Risk**: Medium

**Additional Symbols**: TRXUSDT, DOTUSDT, NEARUSDT, ATOMUSDT, LTCUSDT, BCHUSDT, ARBUSDT, APTUSDT, MATICUSDT, SUIUSDT

```yaml
runtime:
  symbols: [ALL_20_SYMBOLS]
  sniper_mode:
    max_signals_per_hour: 3
    daily_signal_cap: 10
```

**Success Criteria**:
- [ ] Full 20-pair data pipeline stable
- [ ] Memory/CPU usage within limits
- [ ] Prometheus metrics scaling
- [ ] No performance degradation

### Phase 4: Smart Router + TCA Integration
**Duration**: 1 week  
**Caps**: 4/hour, 15/day  
**Risk**: Medium-High

**New Features**:
- Smart Order Routing (S55)
- TCA Feedback Loop (S62)
- Multi-venue execution
- Real-time cost optimization

```yaml
execution:
  smart_order_routing: true
  tca_feedback_enabled: true
  venues: ["BINANCE", "BYBIT", "OKX"]
```

---

## 📊 Monitoring & Thresholds

### Phase-Specific Monitoring

#### Phase 1 (5 Symbols)
| Metric | Threshold | Action |
|--------|-----------|--------|
| Daily P&L | <-2% | Review and pause |
| Slippage Avg | >15 bps | Investigate routing |
| Latency P95 | >500ms | Check infrastructure |
| Error Rate | >1% | Debug and fix |

#### Phase 2 (10 Symbols)  
| Metric | Threshold | Action |
|--------|-----------|--------|
| Daily P&L | <-3% | Portfolio review |
| Memory Usage | >1GB | Optimize or scale |
| Correlation Risk | >0.8 | Reduce exposure |
| Feature Freshness | <95% | Data pipeline check |

#### Phase 3 (20 Symbols)
| Metric | Threshold | Action |
|--------|-----------|--------|
| Daily P&L | <-4% | Risk model review |
| CPU Usage | >60% | Performance optimization |
| Prometheus Load | >80% | Metrics optimization |
| Redis Memory | >500MB | Counter cleanup |

#### Phase 4 (Smart Router)
| Metric | Threshold | Action |
|--------|-----------|--------|
| Router Latency | >200ms | Route optimization |
| Venue Spread | >20 bps | Venue selection review |
| TCA Accuracy | <90% | Model recalibration |
| Multi-venue P&L | Negative vs single | Router evaluation |

---

## 🛡️ Risk Controls & Circuit Breakers

### Phase-Specific Risk Limits

#### Capital at Risk
| Phase | Max Portfolio Risk | Max Single Trade | Max Daily Loss |
|-------|-------------------|------------------|----------------|
| 1 (5 symbols) | 2.5% | 0.5% | 2.0% |
| 2 (10 symbols) | 4.0% | 0.5% | 3.0% |
| 3 (20 symbols) | 6.0% | 0.5% | 4.0% |
| 4 (Smart Router) | 8.0% | 0.5% | 5.0% |

#### Auto-Pause Triggers
```yaml
# Phase-adaptive circuit breakers
circuit_breakers:
  phase_1:
    consecutive_losses: 3
    daily_loss_pct: 2.0
    venue_errors: 5
  phase_2:
    consecutive_losses: 4
    daily_loss_pct: 3.0
    correlation_spike: 0.85
  phase_3:
    consecutive_losses: 5
    daily_loss_pct: 4.0
    memory_usage_mb: 1000
  phase_4:
    consecutive_losses: 5
    daily_loss_pct: 5.0
    router_failures: 10
```

---

## 📈 Success Metrics & KPIs

### Primary KPIs (All Phases)
1. **Uptime**: >99.5%
2. **Signal Accuracy**: >60% win rate
3. **Risk Management**: Daily VaR within target
4. **Execution Quality**: Slippage <20 bps average
5. **System Performance**: Latency P95 <500ms

### Phase-Specific KPIs

#### Phase 1: Stability Focus
- ✅ Zero system crashes
- ✅ Sniper caps 100% enforced
- ✅ MTF confirmation working
- ✅ TCA baseline established

#### Phase 2: Scaling Validation
- ✅ Portfolio correlation managed
- ✅ Resource usage linear scaling
- ✅ Feature store performance maintained
- ✅ Prometheus metrics stable

#### Phase 3: Full Load Testing
- ✅ All 20 symbols operational
- ✅ Memory usage <2GB
- ✅ CPU usage <50% average
- ✅ No feature computation bottlenecks

#### Phase 4: Advanced Features
- ✅ Smart routing saves >5 bps average
- ✅ TCA feedback improves fills
- ✅ Multi-venue execution stable
- ✅ Cost optimization measurable

---

## 🚀 Deployment Procedures

### Phase Transition Checklist

#### Pre-Phase Validation
1. **Previous phase metrics review** ✅
2. **System health check** ✅
3. **Configuration validation** ✅
4. **Rollback plan confirmed** ✅
5. **Monitoring dashboard updated** ✅

#### Phase Deployment Steps
1. **Update configuration files**
2. **Deploy with feature flags** (gradual enable)
3. **Monitor for 1 hour** (enhanced alerting)
4. **Full enable if stable**
5. **24-hour observation period**

#### Phase Validation
1. **Run comprehensive tests**
2. **Validate all success criteria**
3. **Review performance metrics**
4. **Document lessons learned**
5. **Plan next phase or investigate issues**

---

## 🔄 Rollback Procedures

### Immediate Rollback Triggers
- **System crash or hang**
- **Data corruption detected**
- **Daily loss exceeds phase limit**
- **Venue connectivity lost**
- **Redis/infrastructure failure**

### Rollback Execution
```bash
# Emergency rollback to safe state
git checkout v0.9.3-shadow-ready
systemctl restart ultra-signals
# Monitor for 30 minutes before declaring stable
```

### Gradual Rollback (Performance Issues)
1. **Reduce symbol count** to previous phase
2. **Lower sniper caps** by 50%
3. **Disable advanced features** (Smart Router, etc.)
4. **Increase monitoring frequency**
5. **Root cause analysis and fix**

---

## 📅 Timeline & Milestones

### Recommended Schedule
```
Week 1: Shadow → Canary → Phase 1 (5 symbols)
Week 2: Phase 2 (10 symbols) → Phase 3 (20 symbols) 
Week 3: Phase 4 (Smart Router) deployment
Week 4: Full production optimization and fine-tuning
```

### Go/No-Go Decision Points
- **Shadow → Canary**: All tests pass, system stable
- **Canary → Phase 1**: Live execution quality confirmed
- **Phase 1 → Phase 2**: 5-symbol stability proven
- **Phase 2 → Phase 3**: Scaling challenges addressed
- **Phase 3 → Phase 4**: Full load performance validated

---

## 📋 Stakeholder Communication

### Daily Reports (During Rollout)
- **Performance Summary**: P&L, trades, rejections
- **System Health**: Latency, errors, uptime
- **Risk Metrics**: VaR, correlation, exposure
- **Next Steps**: Plan for next 24 hours

### Weekly Reviews
- **Phase Assessment**: Success criteria review
- **Risk Analysis**: Drawdown, correlation changes
- **System Evolution**: Performance trends
- **Strategic Planning**: Long-term optimization

---

**Final Goal**: Full 20-symbol production with Smart Router, TCA feedback, and 4/hour, 15/day sniper caps operating at >99.5% uptime with <20 bps average slippage.

*Generated: 2025-08-28 20:13 UTC*  
*Version: v0.9.3-shadow-ready*  
*Status: Ready for Phase 1 execution*
