
# Canary Run Results

- **Duration**: 2 Hours (Simulated)
- **Pairs**: 20
- **Profile**: Debug Canary

## Signal Generation Summary

| Metric                  | Count |
|-------------------------|-------|
| Candidate Signals       | 15 |
| [OK] Allowed Signals (PRE)  | 2 |
| [BLOCK] Blocked Signals      | 13 |

## Block Reason Histogram

| Veto Reason        | Count |
|--------------------|-------|
| `MTF_DISAGREE` | 5 |
| `VPIN_HIGH` | 3 |
| `REGIME_LOW_CONF` | 4 |
| `CIRCUIT_HALT` | 1 |

## Sample Telegram Messages

### Allowed Signal (PRE)
```
[CHART] *New Ensemble Decision: LONG BTCUSDT* (5m)

Ensemble Confidence: *78.50%*
Vote: `3/4` | Profile: `trend` | Wgt Sum: `0.820`
PRE: p=0.68 | reg=trend | veto=0 | lat_p50=25.1ms p90=45.3ms
--------------------------------------
*Contributing Signals:*
🟢 breakout_v2 (0.85)
🟢 volume_surge (0.80)
🟢 oi_pump (0.75)
🔴 rsi_extreme (-0.60)
```

### Blocked Signal (Debug)
```
[BLOCK] *BLOCKED* — ETHUSDT (5m)
[CHART] *New Ensemble Decision: LONG ETHUSDT* (5m)

Ensemble Confidence: *72.30%*
🚨 *VETOED* — Top reason: `MTF_DISAGREE`
All reasons: `MTF_DISAGREE, REGIME_LOW_CONF`
--------------------------------------
*Contributing Signals:*
🟢 breakout_v2 (0.80)
🟢 volume_surge (0.75)
🔴 rsi_extreme (-0.55)
```

## Summary & Next Steps

The canary run successfully identified several wiring issues, primarily related to regime confidence and funding data resolution. After applying fixes, the system is now generating both allowed and blocked signals, with clear reasons provided via Telegram. The pipeline appears to be functioning as expected.
