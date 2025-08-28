# Shadow Test Results - v0.9.3-shadow-ready

## Test Execution Summary
- **Start Time**: 2025-08-28 20:13 UTC
- **Duration**: 120 minutes (2 hours)
- **Mode**: SHADOW (no live orders)
- **Symbols**: BTCUSDT, ETHUSDT, SOLUSDT
- **Sniper Config**: 2/hour, 6/day, MTF required

---

## 📊 Signal Analysis

### Allowed vs Blocked Signals
```
Total Signal Candidates: [TO BE FILLED DURING TEST]
├── ✅ Allowed Signals: [COUNT] 
├── 🚫 Blocked by Hourly Cap: [COUNT]
├── 🚫 Blocked by Daily Cap: [COUNT]
├── 🚫 Blocked by MTF Confirm: [COUNT]
└── 🚫 Other Rejections: [COUNT]

Expected: 2-4 allowed signals over 2 hours
Actual: [TO BE MEASURED]
```

### Signal Distribution by Symbol
| Symbol | Candidates | Allowed | Hourly Blocked | Daily Blocked | MTF Blocked |
|--------|------------|---------|----------------|---------------|-------------|
| BTCUSDT | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| ETHUSDT | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| SOLUSDT | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

---

## ⚡ Performance Metrics

### Latency Analysis
```
Metric                    | P50    | P95    | P99    | Target | Status
-------------------------|--------|--------|--------|--------|--------
Tick → Decision (ms)     | [TBD]  | [TBD]  | [TBD]  | <500   | [PASS/FAIL]
Decision → Wire (ms)     | [TBD]  | [TBD]  | [TBD]  | <200   | [PASS/FAIL]
Risk Filter Time (ms)    | [TBD]  | [TBD]  | [TBD]  | <50    | [PASS/FAIL]
```

### System Stability
- **Memory Usage**: Start: [TBD] MB → End: [TBD] MB → Growth: [TBD] MB
- **CPU Utilization**: Avg: [TBD]% → Peak: [TBD]%
- **Crashes/Errors**: [COUNT] (Target: 0)
- **Redis Connection**: [STABLE/FALLBACK TO MEMORY]

---

## 🎯 Sniper Enforcement Validation

### Prometheus Metrics Captured
```
# Final counters after 2 hours
sniper_rejections_total{reason="hourly_cap"} = [TBD]
sniper_rejections_total{reason="daily_cap"} = [TBD]  
sniper_rejections_total{reason="mtf_required"} = [TBD]
signals_candidates_total = [TBD]
signals_blocked_total{reason="SNIPER"} = [TBD]
```

### MTF Confirmation Testing
- **MTF Disagreements Detected**: [COUNT]
- **Signals Blocked by MTF**: [COUNT]
- **False Positives** (should be 0): [COUNT]

---

## 📱 Telegram Integration

### Message Examples (Redacted)
```
✅ ALLOWED SIGNAL:
SHADOW | BTCUSDT | LONG | ENTRY:43250 | SL:42800 | TP:44100 | Lev:3x | p:0.65 | regime:trend | veto:none | sniper:allowed

🚫 BLOCKED SIGNAL:
[NO TELEGRAM MESSAGE - CORRECTLY FILTERED]

📊 PRE-TRADE ONLY:
SHADOW | Signal candidate detected but blocked by sniper hourly cap (2/2 used)
```

### Message Quality Check
- ✅ Pre-trade messages only for allowed signals
- ✅ No trade cards sent for blocked signals  
- ✅ Sniper status included in message template
- ✅ All required fields present (entry, SL, TP, etc.)

---

## 🔍 Data Quality Assessment

### Per-Symbol Freshness Check
| Symbol | Data Fresh | Features Fresh | Last Update | Veto Status |
|--------|------------|----------------|-------------|-------------|
| BTCUSDT | [✅/❌] | [✅/❌] | [TIMESTAMP] | [ACTIVE/NONE] |
| ETHUSDT | [✅/❌] | [✅/❌] | [TIMESTAMP] | [ACTIVE/NONE] |
| SOLUSDT | [✅/❌] | [✅/❌] | [TIMESTAMP] | [ACTIVE/NONE] |

### Feature Store Pipeline
- **Collectors → Feature Store**: [STATUS]
- **Feature Store → Veto Stack**: [STATUS] 
- **Veto Stack → Router**: [STATUS]
- **Router → TCA**: [STATUS]

---

## 🛡️ Circuit Breaker Testing

### Circuit Breaker Status
- **Total Triggers**: [COUNT] (Expected: 0 for normal market)
- **Auto-Pause Events**: [COUNT]
- **Recovery Time**: [AVG TIME] seconds
- **False Alarms**: [COUNT] (Target: 0)

### Veto System Performance
| Veto Type | Triggers | Blocks | False Pos | Performance |
|-----------|----------|--------|-----------|-------------|
| VPIN | [TBD] | [TBD] | [TBD] | [PASS/FAIL] |
| Kyle Lambda | [TBD] | [TBD] | [TBD] | [PASS/FAIL] |
| Funding Rate | [TBD] | [TBD] | [TBD] | [PASS/FAIL] |
| News Events | [TBD] | [TBD] | [TBD] | [PASS/FAIL] |

---

## 📋 Exit Criteria Assessment

### ✅ Success Criteria
- [ ] **Caps Enforced**: Hourly (2) and daily (6) limits respected
- [ ] **MTF Blocking**: Signals without confluence properly blocked
- [ ] **No Crashes**: Zero system crashes during 120 minutes
- [ ] **Stable Performance**: Latency p95 < 500ms throughout test
- [ ] **Data Freshness**: All 3 symbols showing fresh data/features
- [ ] **Metrics Working**: Prometheus counters incrementing correctly
- [ ] **Telegram Correct**: Pre-trade only for allowed, no cards for blocked

### Decision Matrix
| Criteria | Status | Notes |
|----------|--------|-------|
| Signal Throttling | [PASS/FAIL] | [DETAILS] |
| MTF Enforcement | [PASS/FAIL] | [DETAILS] |
| System Stability | [PASS/FAIL] | [DETAILS] |
| Monitoring | [PASS/FAIL] | [DETAILS] |
| Integration | [PASS/FAIL] | [DETAILS] |

---

## 🚀 Recommendation

### Shadow Test Result: [PASS/FAIL]

**If PASS → Proceed to Canary Mode**
- Enable BTCUSDT live orders
- Caps: 1/hour, 2/day (more conservative)
- Duration: 60 minutes
- Monitor fills, slippage, trade quality

**If FAIL → Address Issues**
- Review failure details above
- Apply fixes and re-run shadow test
- Consider rollback to v0.9.3-shadow-ready if needed

---

## 📎 Supporting Data

### Log Files
- **Shadow Test Log**: `reports/shadow_test_20250828_201301.log`
- **Prometheus Scrape**: `reports/prometheus_shadow_metrics.json`
- **Grafana Screenshot**: `reports/grafana_shadow_dashboard.png`

### Raw Metrics Export
```json
{
  "test_duration_minutes": 120,
  "signals_processed": "[TBD]",
  "sniper_rejections": {
    "hourly_cap": "[TBD]",
    "daily_cap": "[TBD]", 
    "mtf_required": "[TBD]"
  },
  "performance": {
    "latency_p95_ms": "[TBD]",
    "memory_growth_mb": "[TBD]",
    "cpu_avg_pct": "[TBD]"
  }
}
```

---

*Report Template - To be populated during actual shadow test execution*  
*Generated: 2025-08-28 20:13 UTC*  
*Next Phase: Canary Mode (if shadow passes)*
