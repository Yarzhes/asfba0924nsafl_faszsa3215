# Ultra Signals — Phase-3 Logic Gaps & Fix Plan

## Overview
This document identifies critical issues in the current trading logic implementation and provides a detailed fix plan with file locations, changes, and validation tests.

## A) Feature Correctness Issues

### A1. EMA/ADX/RSI Formula Sanity
**Issue:** Potential look-ahead bias and bar-close misalignment
**Severity:** HIGH
**Files:** `ultra_signals/features/trend.py`, `ultra_signals/features/momentum.py`
**Problems:**
- No explicit verification that calculations use only completed bars
- ADX calculation may include current bar's high/low/close
- RSI calculation window may extend into future

**Fix Plan:**
- Add explicit bar-close verification in feature calculations
- Ensure all indicators use `.iloc[-1]` for latest completed bar only
- Add unit tests with synthetic data to verify no look-ahead

### A2. VWAP/Session VWAP Resets
**Issue:** VWAP may not properly reset at session boundaries
**Severity:** MEDIUM
**Files:** `ultra_signals/features/volume_flow.py`
**Problems:**
- VWAP calculation doesn't account for session boundaries
- Session VWAP not implemented
- No clear session definition in config

**Fix Plan:**
- Implement session-aware VWAP with configurable session boundaries
- Add session VWAP calculation
- Add session boundary detection logic

### A3. Flow Metrics Graceful Degradation
**Issue:** Flow metrics may return NaN instead of neutral values
**Severity:** MEDIUM
**Files:** `ultra_signals/features/flow_metrics.py`
**Problems:**
- Some flow metrics return None/NaN when data unavailable
- No consistent neutral value mapping
- May pollute ensemble with NaN values

**Fix Plan:**
- Ensure all flow metrics return neutral values (0.0) when data unavailable
- Add explicit NaN handling in ensemble combination
- Add unit tests for missing data scenarios

## B) Correlation Double-Counting

### B1. Feature Family Overlap
**Issue:** Highly correlated features may be over-weighted
**Severity:** HIGH
**Files:** `ultra_signals/engine/ensemble.py`
**Problems:**
- EMA cross + MACD histogram both represent trend momentum
- RSI + MACD signal both represent momentum
- No feature family grouping or de-correlation

**Fix Plan:**
- Implement feature family grouping: TREND, MOMENTUM, VOL, FLOW, STRUCTURE
- Sum within family → normalize → weight across families
- Add correlation analysis in ensemble combination

### B2. Multi-TF Feature Correlation
**Issue:** Same features across timeframes may be highly correlated
**Severity:** MEDIUM
**Files:** `ultra_signals/engine/ensemble.py`
**Problems:**
- 1m, 5m, 15m RSI may be highly correlated
- No explicit de-correlation across timeframes
- May lead to over-weighting of similar signals

**Fix Plan:**
- Add correlation-based weighting adjustment
- Implement feature selection based on correlation thresholds
- Add correlation analysis in regime router

## C) Regime Detection Quality

### C1. ADX + EMA Ordering Noise
**Issue:** Current regime detection may be too noisy
**Severity:** HIGH
**Files:** `ultra_signals/engine/regime_router.py`
**Problems:**
- ADX threshold alone may not be sufficient for trend detection
- No secondary confirmation checks
- Missing squeeze detection for range identification

**Fix Plan:**
- **TREND:** Require EMA ladder + ADX ≥ threshold + no squeeze
- **RANGE:** Require ADX low + squeeze or BB width percentile low
- **BREAKOUT:** Require squeeze release + HH/LL break with volume burst
- Add configurable gates in settings

### C2. Multi-TF Confluence Rules
**Issue:** No explicit rules for multi-timeframe agreement
**Severity:** MEDIUM
**Files:** `ultra_signals/engine/confluence.py`
**Problems:**
- Unclear how regime disagreements are resolved
- No weighted relaxation when HTF is neutral
- Missing explicit confluence rules

**Fix Plan:**
- Define explicit confluence rules with configurable weights
- Implement regime agreement scoring
- Add fallback logic for neutral HTF scenarios

## D) Confidence Calibration

### D1. Raw Score to Probability Mapping
**Issue:** Raw weighted sums don't map to actual probabilities
**Severity:** HIGH
**Files:** `ultra_signals/engine/ensemble.py`
**Problems:**
- No calibration of ensemble scores to actual win rates
- Confidence may not reflect true probability
- No historical validation of confidence accuracy

**Fix Plan:**
- Implement logistic calibration: p = 1/(1+e^(−(a+b*x)))
- Fit (a,b) on historical signals with triple-barrier outcomes
- Store calibration coefficients in settings
- Add calibration validation tests

### D2. Confidence Validation
**Issue:** No validation that confidence correlates with actual outcomes
**Severity:** MEDIUM
**Files:** `ultra_signals/engine/ensemble.py`
**Problems:**
- No backtesting of confidence accuracy
- No calibration drift detection
- Missing confidence validation metrics

**Fix Plan:**
- Implement confidence calibration validation
- Add calibration drift detection
- Create confidence accuracy metrics

## E) Anti-Chop & Burst Control

### E1. Non-Maximum Suppression
**Issue:** No suppression of duplicate signals in short time windows
**Severity:** HIGH
**Files:** `ultra_signals/engine/signal_gate.py` (new)
**Problems:**
- Multiple signals for same side within 3-5 minutes
- No signal deduplication logic
- May lead to signal spam

**Fix Plan:**
- Implement NMS per symbol with configurable window
- Keep only highest-confidence signal per side in window
- Add signal deduplication logic

### E2. Flip-Flop Guard
**Issue:** No protection against rapid direction changes
**Severity:** MEDIUM
**Files:** `ultra_signals/engine/signal_gate.py` (new)
**Problems:**
- No minimum distance requirement before reversing
- No cooldown between opposite signals
- May lead to whipsaw trades

**Fix Plan:**
- Require minimum ATR distance before reversing
- Implement cooldown between opposite signals
- Add flip-flop detection and prevention

## F) Exposure & Correlation Caps

### F1. Cluster-Based Exposure Limits
**Issue:** No correlation-based position limits
**Severity:** MEDIUM
**Files:** `ultra_signals/engine/risk_filters.py`
**Problems:**
- No cluster mapping for correlated assets
- No max alerts per cluster enforcement
- Missing correlation-based risk management

**Fix Plan:**
- Add cluster map configuration (BTC/ETH/BNB as one cluster)
- Implement max alerts per cluster
- Add correlation-based exposure limits

### F2. Portfolio Risk Management
**Issue:** No comprehensive portfolio risk limits
**Severity:** MEDIUM
**Files:** `ultra_signals/engine/risk_filters.py`
**Problems:**
- No max concurrent positions limit
- No portfolio-level risk caps
- Missing correlation-based position sizing

**Fix Plan:**
- Add max concurrent positions limit
- Implement portfolio-level risk caps
- Add correlation-based position sizing

## G) Outcome Labeling for Evaluation

### G1. Triple-Barrier Method
**Issue:** No standardized outcome labeling for backtesting
**Severity:** HIGH
**Files:** `ultra_signals/research/labeling.py` (new)
**Problems:**
- No consistent win/loss labeling method
- No triple-barrier implementation
- Missing outcome validation framework

**Fix Plan:**
- Implement triple-barrier method: PT/SL/time
- Create outcome labeling function
- Add outcome validation framework

### G2. Backtest Evaluation
**Issue:** No comprehensive backtest evaluation framework
**Severity:** MEDIUM
**Files:** `ultra_signals/research/backtest_signals.py` (new)
**Problems:**
- No standardized backtest metrics
- No confidence calibration validation
- Missing performance attribution analysis

**Fix Plan:**
- Implement standardized backtest metrics
- Add confidence calibration validation
- Create performance attribution framework

## H) Tick Size & Decimals

### H1. Price Rounding
**Issue:** No proper tick size rounding for entry/SL/TP
**Severity:** MEDIUM
**Files:** `ultra_signals/core/market_meta.py` (new)
**Problems:**
- No tick size metadata handling
- No proper price rounding logic
- Missing exchange-specific tick sizes

**Fix Plan:**
- Implement tick size metadata handling
- Add proper price rounding logic
- Create exchange-specific tick size configuration

## Fix Plan Summary

| Issue | Severity | Files | Change | Test |
|-------|----------|-------|--------|------|
| A1. EMA/ADX/RSI Formula | HIGH | `features/trend.py`, `features/momentum.py` | Add bar-close verification | `test_feature_correctness.py` |
| A2. VWAP Session Resets | MEDIUM | `features/volume_flow.py` | Implement session VWAP | `test_vwap_session.py` |
| A3. Flow Metrics NaN | MEDIUM | `features/flow_metrics.py` | Add neutral value mapping | `test_flow_metrics_graceful.py` |
| B1. Feature Family Overlap | HIGH | `engine/ensemble.py` | Implement family grouping | `test_ensemble_families.py` |
| B2. Multi-TF Correlation | MEDIUM | `engine/ensemble.py` | Add correlation adjustment | `test_correlation_weights.py` |
| C1. Regime Detection Quality | HIGH | `engine/regime_router.py` | Add secondary checks | `test_regime_quality.py` |
| C2. Multi-TF Confluence | MEDIUM | `engine/confluence.py` | Define explicit rules | `test_confluence_rules.py` |
| D1. Confidence Calibration | HIGH | `engine/ensemble.py` | Implement logistic calibration | `test_calibration.py` |
| D2. Confidence Validation | MEDIUM | `engine/ensemble.py` | Add validation metrics | `test_confidence_validation.py` |
| E1. Non-Maximum Suppression | HIGH | `engine/signal_gate.py` | Implement NMS logic | `test_signal_gate_nms.py` |
| E2. Flip-Flop Guard | MEDIUM | `engine/signal_gate.py` | Add flip-flop prevention | `test_flip_flop_guard.py` |
| F1. Cluster Exposure | MEDIUM | `engine/risk_filters.py` | Add cluster limits | `test_cluster_exposure.py` |
| F2. Portfolio Risk | MEDIUM | `engine/risk_filters.py` | Add portfolio caps | `test_portfolio_risk.py` |
| G1. Triple-Barrier | HIGH | `research/labeling.py` | Implement labeling | `test_labeling.py` |
| G2. Backtest Evaluation | MEDIUM | `research/backtest_signals.py` | Add evaluation framework | `test_backtest_evaluation.py` |
| H1. Tick Rounding | MEDIUM | `core/market_meta.py` | Add tick size handling | `test_tick_rounding.py` |

## Implementation Priority

### Phase 1 (Critical - Must Fix)
1. A1. EMA/ADX/RSI Formula Sanity
2. B1. Feature Family Overlap
3. C1. Regime Detection Quality
4. D1. Confidence Calibration
5. E1. Non-Maximum Suppression
6. G1. Triple-Barrier Method

### Phase 2 (Important - Should Fix)
1. A3. Flow Metrics Graceful Degradation
2. B2. Multi-TF Correlation
3. C2. Multi-TF Confluence Rules
4. D2. Confidence Validation
5. E2. Flip-Flop Guard
6. F1. Cluster Exposure Limits
7. H1. Tick Rounding

### Phase 3 (Nice to Have)
1. A2. VWAP Session Resets
2. F2. Portfolio Risk Management
3. G2. Backtest Evaluation

## Validation Criteria

### Unit Tests
- All new functions must have unit tests
- Test edge cases and error conditions
- Verify no look-ahead bias
- Validate graceful degradation

### Integration Tests
- End-to-end signal generation
- Regime detection accuracy
- Ensemble decision consistency
- Risk filter effectiveness

### Backtest Validation
- Confidence calibration accuracy
- Performance improvement vs baseline
- Risk-adjusted returns
- Maximum drawdown control

This gap analysis provides a roadmap for systematically improving the Ultra Signals trading logic while maintaining system stability and performance.



