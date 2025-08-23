# Sprint 7 - Architect Brief

## 1) Config additions (schema only)

```yaml
ensemble:
  enabled: true
  strategies: ["trend_pullback","breakout_book","mean_revert_vwap","sweep_reversal"]
  weights:
    default: {trend_pullback:.30, breakout_book:.30, mean_revert_vwap:.25, sweep_reversal:.15}
    trend:   {trend_pullback:.35, breakout_book:.35, mean_revert_vwap:.15, sweep_reversal:.15}
    mr:      {trend_pullback:.15, breakout_book:.20, mean_revert_vwap:.45, sweep_reversal:.20}
  vote_threshold: 0.55          # weighted confidence to trigger
  veto:
    breakout_requires_book_flip: true
    mr_requires_band_pierce: true

correlation:
  lookbacks: ["1h","4h"]
  refresh_min: 10
  threshold: 0.7
  hysteresis_hits: 2

portfolio:
  max_risk_per_symbol: 0.010     # 1.0% equity
  max_positions_per_symbol: 1
  max_positions_total: 6
  max_cluster_risk: 0.020        # per cluster (long and short tracked separately)
  max_net_long_risk: 0.030
  max_net_short_risk: 0.030
  margin_cap_pct: 0.30           # of equity

brakes:
  min_spacing_sec_same_cluster: 180
  daily_loss_soft_pct: 0.03
  daily_loss_hard_pct: 0.05
  cooldown_after_soft_min: 15
  cooldown_after_streak_symbol: {losses: 3, minutes: 20}
  cooldown_after_streak_global: {losses: 5, minutes: 30}

vol_risk_scale:
  atr_pct_window: 200
  low_vol_pct: 30
  high_vol_pct: 70
  low_vol_boost: 1.20
  high_vol_cut: 0.70
```

## 2) Data models

**SubSignal**

```
{ ts:int, symbol:str, tf:str, strategy_id:str,
  direction:"LONG"|"SHORT"|"FLAT",
  confidence_calibrated: float, reasons: dict }
```

**EnsembleDecision**

```
{ ts:int, symbol:str, tf:str,
  decision:"LONG"|"SHORT"|"FLAT",
  confidence: float,
  subsignals: list[SubSignal],
  vote_detail:{weighted_sum:float, agree:int, total:int, profile:str},
  vetoes:list[str] }
```

**PortfolioState**

```
{ ts:int, equity:float,
  positions: dict[symbol]->{side:str, size:float, entry:float, risk:float, cluster:str},
  exposure:{
    symbol: dict[symbol]->{long_risk:float, short_risk:float},
    cluster: dict[cluster]->{long_risk:float, short_risk:float},
    net:{long:float, short:float}, margin_used:float } }
```

**RiskEvent**

```
{ ts:int, symbol:str, reason:str, action:"VETO"|"DOWNSIZE"|"COOLDOWN", detail:dict }
```

## 3) Module interfaces

**strategies/\*.py** (one file per strategy)

* `def generate_subsignal(fv, ctx) -> SubSignal`

**engine/ensemble.py**

* `def combine_subsignals(subs:list[SubSignal], regime_profile:str, settings) -> EnsembleDecision`

**analytics/correlation.py**

* `def compute_corr_groups(returns_df, threshold:float) -> dict[str,str]`  # symbol->cluster
* `def update_corr_state(prev_state, new_groups, hysteresis_hits:int) -> dict[str,str]`

**risk/portfolio.py**

* `def evaluate_portfolio(decision:EnsembleDecision, state:PortfolioState, settings) -> tuple[bool, float, list[RiskEvent]]`

  * Returns `(allowed, size_scale, events)` where `size_scale` may be <1 to downsize.

**engine/sizing.py** (update)

* `def apply_volatility_scaling(base_risk:float, atr_percentile:float, cfg) -> float`

**engine/risk\_filters.py** (update)

* Accept `portfolio_ctx` and append **RiskEvents** for spacing, daily loss brakes, streak cooldowns.

**backtest/event\_runner.py** (update)

* Use portfolio evaluation to gate/resize trades; write RiskEvents.

**transport/telegram.py** (update)

* Include ensemble vote summary + top veto reason (optional).

## 4) Flow (sequence)

1. Build all **SubSignals** per strategy.
2. **Ensemble** merges → `EnsembleDecision`.
3. **Portfolio** evaluation (exposure, brakes, spacing) → allow/resize/veto.
4. If allowed → position size (with **volatility scaling**) → signal/entry path.
5. Log **RiskEvents** and vote details.

## 5) Performance budgets

* Correlation refresh async every `refresh_min` minutes; off hot path.
* Ensemble combine ≤ 2 ms per symbol/tf.
* Portfolio checks O(1) per open position & cluster.

## 6) Test plan (names)

* `test_subsignals_per_strategy.py`
* `test_ensemble_weighted_vote_and_veto.py`
* `test_correlation_groups_hysteresis.py`
* `test_portfolio_exposure_caps_and_netting.py`
* `test_brakes_spacing_daily_loss_streaks.py`
* `test_volatility_scaled_sizing.py`
* `test_backtest_portfolio_integration.py`
* `test_telegram_vote_summary.py`

## 7) Acceptance checklist

* Config validates with defaults; turning `ensemble.enabled=false` falls back to single-engine path.
* Ensemble outputs 1 decision with clear reasons; vetoes logged.
* Portfolio caps & brakes enforce as configured in both **live** and **backtest**.
* Walk-forward runs with ensemble + portfolio, producing KPIs per strategy and overall.