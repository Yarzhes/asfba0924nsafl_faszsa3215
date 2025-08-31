# Ultra Signals — Phase-3 Logic Specification

## Overview
This document specifies the end-to-end trading logic of the Ultra Signals system, covering how features are computed, regimes are detected, ensemble decisions are made, and risk management is applied.

## 1. Data/Features Used (by Module)

### 1.1 Trend Features (`ultra_signals/features/trend.py`)
**Inputs:** OHLCV DataFrame
**Computation:**
- **EMA Short/Medium/Long:** Exponential Moving Averages with periods 20/50/200 (configurable via `features.trend.ema_short/medium/long`)
- **ADX:** Average Directional Index with period 14 (configurable via `features.trend.adx_period`)
- **Bar-close alignment:** All calculations use close prices from completed bars
- **Window requirements:** Minimum data = max(ema_short, ema_medium, ema_long) periods

### 1.2 Momentum Features (`ultra_signals/features/momentum.py`)
**Inputs:** OHLCV DataFrame
**Computation:**
- **RSI:** Relative Strength Index with period 14 (configurable via `features.momentum.rsi_period`)
- **MACD:** Moving Average Convergence Divergence with fast=12, slow=26, signal=9 (configurable via `features.momentum.macd_fast/slow/signal`)
- **Bar-close alignment:** Uses close prices from completed bars
- **Window requirements:** RSI needs rsi_period+1, MACD needs slow+signal periods

### 1.3 Volatility Features (`ultra_signals/features/volatility.py`)
**Inputs:** OHLCV DataFrame
**Computation:**
- **ATR:** Average True Range with period 14 (configurable via `features.volatility.atr_period`)
- **ATR Percentile:** Rolling percentile over 200 bars (configurable via `features.volatility.atr_percentile_window`)
- **Bollinger Bands:** Upper/lower bands with period 20, stddev 2.0 (configurable via `features.volatility.bbands_period/stddev`)
- **Bar-close alignment:** Uses high/low/close from completed bars

### 1.4 Volume Flow Features (`ultra_signals/features/volume_flow.py`)
**Inputs:** OHLCV DataFrame
**Computation:**
- **VWAP:** Volume Weighted Average Price with window 20 (configurable via `features.volume_flow.vwap_window`)
- **VWAP Std Devs:** Standard deviation bands at 1.0 and 2.0 (configurable via `features.volume_flow.vwap_std_devs`)
- **Volume Z-Score:** Z-score of volume over window 20 (configurable via `features.volume_flow.volume_z_window`)

### 1.5 Flow Metrics (`ultra_signals/features/flow_metrics.py`)
**Inputs:** OHLCV, trades, liquidations, orderbook top
**Computation:**
- **CVD (Cumulative Volume Delta):** Buy volume - sell volume per bar
- **OFI (Order Flow Imbalance):** (bid_size - ask_size) / (bid_size + ask_size)
- **Liquidation Events:** Clustered liquidation notional with ATR normalization
- **Depth Imbalance:** Cross-venue depth comparison
- **Graceful degradation:** Returns None/neutral values when data unavailable

### 1.6 Derivatives Features (`ultra_signals/features/derivatives_posture.py`)
**Inputs:** Funding rates, OI data, liquidation data
**Computation:**
- **Funding Rate:** Current rate, trailing 8 periods, prediction (configurable via `derivatives.funding_trail_len`)
- **OI Changes:** 1m/5m/15m deltas with z-score normalization
- **Liquidation Pulse:** Notional-weighted liquidation clusters (configurable via `derivatives.liq_pulse.window_sec`)

## 2. Regime Detection

### 2.1 Current Implementation (`ultra_signals/engine/regime_router.py`)
**Primary Method:** Uses existing regime classifier if available, falls back to heuristics
**Heuristic Rules:**
- **TREND:** ADX ≥ 22 (configurable via `regime_detection.adx_threshold`)
- **CHOP:** ATR percentile ≤ 15% (configurable via `regime_detection.chop_volatility`)
- **MEAN_REVERT:** RSI ≥ 70 or ≤ 30 (configurable via `regime_detection.mean_revert_rsi`)
- **Default:** "mixed" regime

### 2.2 Multi-TF Confluence
**Configuration:** `confluence.map` defines confirmation timeframes
- Entry TF (1m/5m) → confirmation TF (15m) → regime TF (1h/4h)
- **Requirement:** At least 2/3 timeframes must agree (configurable via `confluence.require_regime_align`)

### 2.3 Regime Disagreement Handling
**Current:** Returns "mixed" regime with default alpha profile
**Proposed:** Weight-based relaxation when HTF is neutral

## 3. Ensemble Decision

### 3.1 Input Processing (`ultra_signals/engine/ensemble.py`)
**Subsignals:** List of SubSignal objects from different alpha strategies
**Normalization:** 
- Raw scores mapped to [-1, 1] range
- Probabilities mapped to [0, 1] range
- NaN values → neutral (0.0)

### 3.2 Weighted Voting
**Profile-based weights:** Different weight schemes per regime (configurable via `weights_profiles`)
- **Trend:** trend(0.3), momentum(0.22), volatility(0.08), flow(0.18), orderbook(0.08), derivatives(0.14)
- **Mean Revert:** trend(0.15), momentum(0.22), volatility(0.13), flow(0.22), orderbook(0.13), derivatives(0.15)
- **Chop:** trend(0.1), momentum(0.2), volatility(0.2), flow(0.25), orderbook(0.15), derivatives(0.1)

### 3.3 Decision Thresholds
**Vote Threshold:** Configurable per regime (default 0.45, configurable via `ensemble.vote_threshold`)
**Margin of Victory:** Minimum 0.08 separation (configurable via `ensemble.margin_of_victory`)
**Confidence Floor:** Minimum 0.65 confidence (configurable via `ensemble.confidence_floor`)

### 3.4 Tie-breakers and Confidence
**Tie-breaker:** Higher confidence wins
**Confidence calculation:** Weighted average of subsignal confidences
**Probability mass:** Uses calibrated probabilities when available (configurable via `ensemble.use_prob_mass`)

## 4. Risk Stack and Vetoes

### 4.1 Veto Order (`ultra_signals/engine/quality_gates.py`)
1. **Data Ready:** Sufficient bars, no large gaps
2. **Spread:** Max spread percentage (configurable via `quality_gates.veto.max_spread_pct`)
3. **Slippage/ATR%:** ATR percentage limit (configurable via `quality_gates.veto.atr_pct_limit`)
4. **ADX Gate:** Minimum ADX threshold (configurable via `quality_gates.veto.adx_min`)
5. **News Veto:** High-impact news embargo (configurable via `news_veto`)
6. **Exposure Cap:** Portfolio risk limits (configurable via `portfolio_risk`)

### 4.2 Veto Visibility
**Logging:** Single-line INFO with key metrics for vetoed signals
**Reason codes:** Specific veto reason included in decision object

## 5. Position/Risk Sizing

### 5.1 Risk Calculation (`ultra_signals/engine/position_sizer.py`)
**Base Risk:** 1% per trade (configurable via `position_sizing.max_risk_pct`)
**ATR-based sizing:** Position size = risk_amount / (ATR * multiplier)
**Stop Loss:** ATR * 1.5 multiplier (configurable via `execution.sl_atr_multiplier`)

### 5.2 Take Profit Levels
**TP1:** 1.0R (1:1 risk-reward)
**TP2:** 1.5R (1:1.5 risk-reward)
**TP3:** 2.0R (1:2 risk-reward)
**Configurable:** via `execution.tp_levels`

### 5.3 Exposure Limits
**Per Symbol:** Max 20% position value (configurable via `sizer.per_symbol.max_position_value_pct`)
**Per Cluster:** Max cluster risk percentage (configurable via `portfolio_risk.max_cluster_risk_pct`)
**Gross Risk:** Max 2.5% total portfolio risk (configurable via `portfolio_risk.max_gross_risk_pct`)

## 6. Notification Contract

### 6.1 Telegram Message Format (`ultra_signals/transport/telegram.py`)
**Required Fields:**
- Symbol and timeframe
- Decision (LONG/SHORT/FLAT)
- Entry price (current market price)
- Stop loss (ATR-based calculation)
- Take profit levels (TP1, TP2, TP3)
- Risk-reward ratio
- Leverage (default 10x)
- Risk percentage (1% default)
- Timestamp
- Confidence score

### 6.2 Price Rounding
**Tick Size:** Rounds to exchange tick size when available
**Fallback:** Uses safe decimals per symbol (configurable via `formatting.tick_size_overrides`)

### 6.3 Message Template
**Format:** `{pair} | {side} | ENTRY:{entry} | SL:{sl} | TP:{tp} | Lev:{lev} | p:{p_win:.2f} | regime:{regime} | veto:{veto_flags} | code:{reason}`
**Configurable:** via `transport.telegram.message_template`

## 7. Configuration Structure

### 7.1 Key Settings Blocks
- **features:** Feature calculation parameters
- **regime:** Regime detection thresholds
- **ensemble:** Voting and confidence parameters
- **quality_gates:** Veto thresholds and conditions
- **position_sizing:** Risk and sizing parameters
- **execution:** SL/TP and leverage settings
- **transport:** Notification formatting

### 7.2 Default Values
All parameters have sensible defaults that can be overridden via settings.yaml
Critical parameters are documented with their default values in this specification

## 8. Data Flow Summary

1. **Data Collection:** OHLCV bars, trades, orderbook, funding rates
2. **Feature Computation:** Technical indicators, flow metrics, derivatives
3. **Regime Detection:** Multi-timeframe regime classification
4. **Alpha Generation:** Subsignals from different strategies
5. **Ensemble Voting:** Weighted combination with confidence calibration
6. **Risk Filtering:** Veto stack with specific reason codes
7. **Position Sizing:** ATR-based risk calculation
8. **Notification:** Formatted Telegram message with trade details

This specification provides the foundation for auditing and upgrading the trading logic while maintaining the existing system architecture.



