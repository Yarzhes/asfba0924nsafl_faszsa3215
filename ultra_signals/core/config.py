"""
Configuration loader for Ultra-Signals.

This module provides Pydantic models for strong validation of settings
and a loader function that merges a YAML configuration file with
environment variables.

Design Principles:
- Strict Schema: All settings are defined in Pydantic models to ensure type
  safety and validate constraints (e.g., value ranges, list contents).
- Environment Overrides: Any setting can be overridden by an environment
  variable. This is useful for CI/CD, Docker, and keeping secrets out of
  the YAML file. The override mechanism follows a nested structure, e.g.,
  `transport.telegram.bot_token` can be overridden by the environment
  variable `ULTRA_SIGNALS_TRANSPORT__TELEGRAM__BOT_TOKEN`.
- Single Source of Truth: The `load_settings` function is the single entry
  point for accessing configuration, returning an immutable `Settings` object.
- Clear Errors: If validation fails, Pydantic raises a detailed `ValidationError`
  which is wrapped in a custom `ConfigError` for clear, actionable feedback.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_core import PydanticCustomError

# --- Custom Exceptions ---

class ConfigError(Exception):
    """Custom exception for configuration-related errors."""
    pass

# --- Pydantic Models for Configuration Sections ---

class DataSourceSettings(BaseModel):
    """Settings for data providers like Binance."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

class RuntimeSettings(BaseModel):
    """Core runtime settings for the application."""
    symbols: List[str] = Field(..., min_length=1)
    timeframes: List[str] = Field(..., min_length=1)
    primary_timeframe: str
    reconnect_backoff_ms: int = Field(gt=0)

    @field_validator('primary_timeframe')
    def primary_timeframe_must_be_in_timeframes(cls, v, values):
        if 'timeframes' in values.data and v not in values.data['timeframes']:
            raise PydanticCustomError(
                "primary_timeframe_invalid",
                "Primary timeframe '{primary_timeframe}' must be one of the subscribed timeframes: {timeframes}",
                {"primary_timeframe": v, "timeframes": values.data['timeframes']}
            )
        return v

class TrendFeatureSettings(BaseModel):
    """Parameters for trend features."""
    ema_short: int = Field(gt=0)
    ema_medium: int = Field(gt=0)
    ema_long: int = Field(gt=0)

class MomentumFeatureSettings(BaseModel):
    """Parameters for momentum features."""
    rsi_period: int = Field(gt=0)
    macd_fast: int = Field(gt=0)
    macd_slow: int = Field(gt=0)
    macd_signal: int = Field(gt=0)

class VolatilityFeatureSettings(BaseModel):
    """Parameters for volatility features."""
    atr_period: int = Field(gt=0)
    bbands_period: int = Field(gt=0)
    bbands_stddev: float = Field(gt=0)

class VolumeFlowFeatureSettings(BaseModel):
    """Parameters for volume and VWAP features."""
    vwap_window: int = Field(20, gt=0)
    vwap_std_devs: Tuple[float, ...] = (1.0, 2.0)
    volume_z_window: int = Field(50, gt=0)

class WhaleFeatureSettings(BaseModel):
    """Sprint 41 Whale / Smart Money feature settings.

    All windows in seconds. Enabled flag gates computation in FeatureStore.
    Multipliers used for burst detection.
    """
    enabled: bool = False
    short_sec: int = Field(3600, gt=0)
    medium_sec: int = Field(6*3600, gt=0)
    long_sec: int = Field(24*3600, gt=0)
    deposit_burst_multiplier: float = Field(3.0, ge=1.0)
    withdrawal_burst_multiplier: float = Field(3.0, ge=1.0)
    min_block_notional_usd: float = Field(500_000.0, ge=0)
    min_sweep_notional_usd: float = Field(1_000_000.0, ge=0)
    options_enabled: bool = True
    smart_money_enabled: bool = True
    # Per-symbol overrides (symbol -> value)
    block_notional_overrides: Dict[str, float] = Field(default_factory=dict)
    sweep_notional_overrides: Dict[str, float] = Field(default_factory=dict)
    deposit_burst_multiplier_overrides: Dict[str, float] = Field(default_factory=dict)
    withdrawal_burst_multiplier_overrides: Dict[str, float] = Field(default_factory=dict)

class WhaleRiskSettings(BaseModel):
    """Map whale feature flags / conditions to veto or boost actions.

    action: VETO | DAMPEN | BOOST (BOOST -> size multiplier >1 downstream)
    size_mult applies for DAMPEN / BOOST.
    """
    enabled: bool = True
    deposit_spike_action: Literal['VETO','DAMPEN','NONE'] = 'DAMPEN'
    withdrawal_spike_action: Literal['BOOST','DAMPEN','NONE','VETO'] = 'BOOST'
    block_sell_extreme_action: Literal['VETO','DAMPEN','NONE'] = 'VETO'
    block_buy_extreme_action: Literal['BOOST','NONE','DAMPEN'] = 'BOOST'
    composite_pressure_boost_thr: float = 5_000_000.0
    composite_pressure_veto_thr: float = -5_000_000.0
    boost_size_mult: float = 1.25
    dampen_size_mult: float = 0.7

class FeatureSettings(BaseModel):
    """Container for all feature computation parameters."""
    warmup_periods: int = Field(gt=1)
    trend: TrendFeatureSettings
    momentum: MomentumFeatureSettings
    volatility: VolatilityFeatureSettings
    volume_flow: VolumeFlowFeatureSettings
    whales: WhaleFeatureSettings = WhaleFeatureSettings()
    whale_risk: WhaleRiskSettings = WhaleRiskSettings()

class ScoringWeights(BaseModel):
    """Weights for scoring components."""
    trend: float = Field(ge=0, le=1)
    momentum: float = Field(ge=0, le=1)
    volatility: float = Field(ge=0, le=1)
    flow: float = Field(ge=0, le=1)
    orderbook: float = Field(ge=0, le=1)
    derivatives: float = Field(ge=0, le=1)

class OISettings(BaseModel):
    """Settings for Open Interest provider."""
    enabled: bool = True
    provider: Literal["mock", "coinglass", "coinalyze"] = "mock"
    refresh_sec: int = Field(60, gt=0)

class LiqPulseSettings(BaseModel):
    """Settings for Liquidation Pulse feature."""
    window_sec: int = Field(300, gt=0)
    notional_weight: float = Field(1.0, ge=0)

class DerivativesSettings(BaseModel):
    """Container for all derivatives-related settings."""
    funding_trail_len: int = Field(8, gt=0)
    funding_refresh_sec: int = Field(900, gt=0)
    oi: OISettings
    liq_pulse: LiqPulseSettings


class SentimentSourceWeights(BaseModel):
    influencer: float = Field(3.0, ge=0)
    twitter: float = Field(1.5, ge=0)
    reddit: float = Field(1.0, ge=0)
    telegram: float = Field(0.8, ge=0)
    news: float = Field(0.7, ge=0)


class SentimentSettings(BaseModel):
    enabled: bool = True
    cache_dir: Optional[str] = ".cache/sentiment"
    symbols: List[str] = Field(default_factory=list)
    sources: Dict[str, Any] = Field(default_factory=dict)
    source_weights: SentimentSourceWeights = SentimentSourceWeights()
    influencers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    topic_taxonomy_path: Optional[str] = None  # override path for topic taxonomy file
    funding_threshold: float = Field(0.0003, ge=0.0)
    oi_threshold_pct: float = Field(0.02, ge=0.0)
    extremes: Dict[str, Any] = Field(default_factory=dict)
    telemetry: Dict[str, Any] = Field(default_factory=dict)

class RegimeSettings(BaseModel):
    """Parameters for the market regime classification engine."""
    hysteresis_bars: int = Field(3, ge=0)
    adx_min_trend: int = Field(15, ge=0, le=100)
    atr_percentile_windows: int = Field(200, gt=1)
    variance_ratio_window: int = Field(20, gt=1)
    # Sprint 43 Meta-Regime engine tunables (kept optional / defaulted to avoid breaking existing configs)
    enabled: bool = True
    regimes: List[str] = Field(default_factory=lambda: [
        "trend_up","trend_down","chop_lowvol","panic_deleverage","gamma_pin","carry_unwind","risk_on","risk_off"
    ])
    clusterer: Dict[str, Any] = Field(default_factory=lambda: {
        "method": "hdbscan",              # hdbscan | kmeans
        "min_cluster_size": 40,
        "min_samples": 10,
        "k": 8
    })
    hmm: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "n_states": 8,
        "covariance_type": "diag"
    })
    supervised: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": False,
        "target": "next_vol_spike",       # next_vol_spike | drawdown | trend_strength
        "model": "xgboost",               # xgboost|lgbm|rf|logit
        "calibration": "isotonic"
    })
    change_point: Dict[str, Any] = Field(default_factory=lambda: {
        "method": "bocpd",                 # bocpd | cusum
        "hazard_lambda": 250,
        "cusum_thr": 3.0
    })
    smoothing: Dict[str, Any] = Field(default_factory=lambda: {
        "stickiness": 0.85,                 # probability mass retained on current regime
        "hazard_weight": 0.30
    })
    # Confidence bands for mapping max_prob -> policy flag
    confidence_bands: Dict[str, float] = Field(default_factory=lambda: {
        "high": 0.7,
        "medium": 0.4
    })
    feature_weights: Dict[str, float] = Field(default_factory=lambda: {
        "price_vol": 0.30,
        "sentiment": 0.15,
        "whales": 0.20,
        "cross_asset": 0.20,
        "positioning": 0.15
    })
    policy_map_path: str = Field("regime_policy_map.json", description="Path to JSON mapping regimes to sizing/veto policies")
    retrain_interval_hours: int = Field(24*7, gt=0)
    min_regime_duration_bars: int = Field(6, ge=1)
    hazard_flip_threshold: float = Field(0.55, ge=0, le=1.0)
    exp_vol_horizon_bars: int = Field(12, gt=0)

class WeightsProfilesSettings(BaseModel):
    """Defines different scoring weight profiles for market regimes."""
    trend: ScoringWeights
    mean_revert: ScoringWeights
    chop: ScoringWeights

class FiltersSettings(BaseModel):
    """Global filters for signal generation."""
    avoid_funding_minutes: int = Field(5, ge=0)

class ThresholdSettings(BaseModel):
    """Thresholds for generating signals."""
    enter: float = Field(ge=0, le=1)
    exit: float = Field(ge=0, le=1)

class RiskSettings(BaseModel):
    """Risk management settings."""
    # Max spread as a percentage. Can be a global default or overridden per symbol.
    max_spread_pct: Dict[str, float] = Field(default_factory=lambda: {"default": 0.05})
    # Note: funding avoidance moved to top-level `filters`.

class EngineSettings(BaseModel):
    """Container for all engine-related settings."""
    scoring_weights: Dict[str, float]  # Deprecated, kept for schema compatibility
    thresholds: ThresholdSettings
    risk: RiskSettings

class EnsembleSettings(BaseModel):
    """Settings for the strategy ensemble."""
    min_score: float = Field(0.05, ge=0, le=1)
    majority_threshold: float = Field(0.50, ge=0.5, le=1.0)
    # Veto rule toggles
    veto_trend_flip: bool = True
    veto_band_pierce: bool = True

class CorrelationSettings(BaseModel):
    """Settings for the correlation engine."""
    enabled: bool = True
    lookback_periods: int = Field(200, gt=1)
    cluster_threshold: float = Field(0.7, ge=0, le=1.0)
    hysteresis: float = Field(0.1, ge=0, le=1.0)
    refresh_interval_bars: int = Field(50, gt=1)

class PortfolioSettings(BaseModel):
    """
    Settings for portfolio-level risk management.

    NOTE: These fields include the original exposure limits plus the newer
    position-count and scale-in controls so YAML keys like
    `max_positions_per_symbol` and `allow_scale_in` are no longer ignored.
    """
    # Original exposure-style limits
    max_exposure_per_symbol: float = Field(1000.0, gt=0)
    max_exposure_per_cluster: float = Field(3000.0, gt=0)
    max_net_exposure: float = Field(5000.0, gt=0)
    max_margin_pct: float = Field(50.0, gt=0, le=100.0)
    max_total_positions: int = Field(10, gt=0)

    # Newer position-count / scale-in controls (previously missing from schema)
    max_positions_total: int = Field(10, gt=0, description="Max open positions across all symbols (alias for max_total_positions if your engine uses it).")
    max_positions_per_symbol: int = Field(1, gt=0, description="How many concurrent positions allowed per symbol.")
    allow_scale_in: bool = Field(False, description="Allow additional entries while a position is already open.")
    scale_in_cooldown_bars: int = Field(0, ge=0, description="Minimum bars to wait before adding again.")
    scale_in_min_distance_pct: float = Field(0.0, ge=0.0, description="Minimum price move (fraction) between adds, e.g. 0.0025 = 0.25%.")


# --- Sprint 22 Portfolio Hedge Settings ---
class BetaBand(BaseModel):
    min: float
    max: float

class HedgeRebalanceSettings(BaseModel):
    min_notional: float = Field(0.005, ge=0)  # fraction of equity
    cooloff_bars: int = Field(3, ge=0)

class HedgeCostsSettings(BaseModel):
    taker_fee: float = Field(0.0004, ge=0)
    slippage_model: str = "atr"
    funding_penalty_perc_per_day: float = Field(0.03, ge=0)

class LeaderBiasSettings(BaseModel):
    trend_up_beta_target: float = 0.05
    trend_down_beta_target: float = -0.05

class OpenGuardSettings(BaseModel):
    block_if_exceeds_beta_cap: bool = True
    downscale_if_over_band: bool = True
    downscale_factor: float = Field(0.5, ge=0, le=1)

class PortfolioHedgeSettings(BaseModel):
    enabled: bool = True
    leader: str = "BTCUSDT"
    timeframe: str = "5m"
    lookback_bars: int = Field(288, gt=1)
    shrinkage_lambda: float = Field(0.1, ge=0, le=1)
    corr_threshold_high: float = Field(0.55, ge=0, le=1)
    beta_band: BetaBand = BetaBand(min=-0.15, max=0.15)
    beta_hard_cap: float = Field(0.25, ge=0)
    rebalance: HedgeRebalanceSettings = HedgeRebalanceSettings()
    costs: HedgeCostsSettings = HedgeCostsSettings()
    cluster_map: Dict[str, str] = Field(default_factory=dict)
    cluster_caps: Dict[str, float] = Field(default_factory=dict)
    leader_bias: LeaderBiasSettings = LeaderBiasSettings()
    open_guard: OpenGuardSettings = OpenGuardSettings()

class BrakesSettings(BaseModel):
    """Settings for emergency brakes and trade spacing."""
    min_spacing_sec_cluster: int = Field(300, ge=0)
    daily_loss_soft_limit_pct: float = Field(2.0, gt=0, le=100.0)
    daily_loss_hard_limit_pct: float = Field(4.0, gt=0, le=100.0)
    streak_cooldown_trades: int = Field(3, ge=0)
    streak_cooldown_hours: int = Field(12, ge=0)

class SizingSettings(BaseModel):
    """Settings for position sizing."""
    vol_risk_scale_pct: float =  Field(0.5, gt=0, le=100.0)

class TelegramSettings(BaseModel):
    """Settings for Telegram transport."""
    enabled: bool = True
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None

class TransportSettings(BaseModel):
    """Container for all transport-related settings."""
    telegram: TelegramSettings
    dry_run: bool = True

class FundingProviderSettings(BaseModel):
    """Settings for the funding rate provider."""
    refresh_interval_minutes: int = Field(15, gt=0)

class BacktestDataSettings(BaseModel):
    """Settings for backtest data sources."""
    provider: Literal["csv", "parquet", "exchange"] = "csv"
    base_path: Optional[str] = None  # For csv/parquet
    cache_path: str = ".cache/data"

class BacktestExecutionSettings(BaseModel):
    """Settings for trade execution simulation."""
    initial_capital: float = Field(10000.0, gt=0)
    default_size_pct: float = Field(1.0, gt=0, le=100)
    fee_bps: float = Field(4.0, ge=0)
    slippage_model: Literal["atr", "book_proxy", "none"] = "atr"

class BacktestSettings(BaseModel):
    """Container for all backtesting-specific settings."""
    start_date: str
    end_date: str
    data: BacktestDataSettings
    execution: BacktestExecutionSettings

class WalkforwardSettings(BaseModel):
    """Settings for walk-forward analysis."""
    train_days: int = Field(90, gt=0)
    test_days: int = Field(30, gt=0)
    warmup_days: int = Field(7, ge=0)
    purge_days: int = Field(3, ge=0)
    embargo_pct: float = Field(0.01, ge=0, le=1.0)

class CalibrationSettings(BaseModel):
    """Settings for confidence calibration."""
    method: Literal["isotonic", "platt"] = "isotonic"
    per_regime: bool = False

class ReportsSettings(BaseModel):
    """Settings for generating reports."""
    output_dir: str = "reports"

class LoggingSettings(BaseModel):
    """Settings for logging configuration."""
    level: str = Field("INFO", description="The logging level, e.g., DEBUG, INFO, WARNING.")

class LiveLatencyBudget(BaseModel):
    target: int = 80
    p99: int = 180

class LiveLatencySettings(BaseModel):
    tick_to_decision_ms: LiveLatencyBudget = LiveLatencyBudget()
    decision_to_order_ms: LiveLatencyBudget = LiveLatencyBudget(target=70, p99=160)

class LiveQueueSettings(BaseModel):
    feed: int = 2000
    engine: int = 256
    orders: int = 128

class LiveRetrySettings(BaseModel):
    max_attempts: int = 4
    base_delay_ms: int = 120

class LiveOrderErrorBurst(BaseModel):
    count: int = 6
    window_sec: int = 120

class LiveCircuitBreakers(BaseModel):
    """Legacy circuit breaker settings (maintained for backwards compatibility)."""
    daily_loss_limit_pct: float = 0.06
    max_consecutive_losses: int = 4
    order_error_burst: LiveOrderErrorBurst = LiveOrderErrorBurst()
    data_staleness_ms: int = 2500

# Sprint 65 - Extreme Event Protection Settings
class ShockDetectionSettings(BaseModel):
    """Configuration for shock detection (Sprint 65)."""
    enabled: bool = True
    return_windows_sec: List[float] = Field(default_factory=lambda: [1.0, 2.0, 5.0])
    warn_k_sigma: float = 4.0
    derisk_k_sigma: float = 5.0
    flatten_k_sigma: float = 6.0
    halt_k_sigma: float = 8.0
    
    rv_horizon_sec: float = 10.0
    rv_warn_z: float = 2.5
    rv_derisk_z: float = 3.0
    rv_flatten_z: float = 4.0
    
    spread_warn_z: float = 2.0
    spread_derisk_z: float = 3.0
    depth_warn_drop_pct: float = 0.5
    depth_derisk_drop_pct: float = 0.7
    
    vpin_warn_pctl: float = 0.90
    vpin_derisk_pctl: float = 0.95
    vpin_flatten_pctl: float = 0.98
    lambda_warn_z: float = 2.0
    lambda_derisk_z: float = 3.0
    
    oi_dump_warn_pct: float = 0.10
    oi_dump_derisk_pct: float = 0.20
    funding_swing_warn_bps: float = 10.0
    funding_swing_derisk_bps: float = 25.0
    
    venue_health_warn: float = 0.8
    venue_health_derisk: float = 0.6
    stablecoin_depeg_warn_bps: float = 20.0
    stablecoin_depeg_derisk_bps: float = 50.0
    
    min_triggers_warn: int = 1
    min_triggers_derisk: int = 2
    min_triggers_flatten: int = 2
    min_triggers_halt: int = 3

class CircuitBreakerPolicySettings(BaseModel):
    """Configuration for circuit breaker policy (Sprint 65)."""
    enabled: bool = True
    warn_threshold: float = 1.0
    derisk_threshold: float = 2.0
    flatten_threshold: float = 3.0
    halt_threshold: float = 4.0
    
    warn_exit_threshold: float = 0.5
    derisk_exit_threshold: float = 1.0
    flatten_exit_threshold: float = 1.5
    halt_exit_threshold: float = 2.0
    
    warn_cooldown_bars: int = 3
    derisk_cooldown_bars: int = 5
    flatten_cooldown_bars: int = 10
    halt_cooldown_bars: int = 20
    
    enable_staged_recovery: bool = True
    recovery_stages: List[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    recovery_stage_bars: int = 5
    
    # Action overrides per level
    warn_size_mult: float = 0.5
    warn_leverage_cap: Optional[float] = 5.0
    derisk_size_mult: float = 0.0
    derisk_leverage_cap: Optional[float] = 3.0
    flatten_size_mult: float = 0.0
    flatten_leverage_cap: Optional[float] = 1.0
    halt_size_mult: float = 0.0

class ExtremeEventAlertsSettings(BaseModel):
    """Configuration for extreme event Telegram alerts (Sprint 65)."""
    enabled: bool = True
    rate_limit_sec: int = 30
    max_triggers_shown: int = 3
    include_countdown: bool = True
    include_technical_details: bool = True

class SafeExitSettings(BaseModel):
    """Configuration for safe position exits (Sprint 65)."""
    max_participation_rate: float = 0.1
    slice_duration_sec: int = 30
    max_slices: int = 10
    passive_timeout_sec: int = 120
    market_urgency_threshold: float = 5.0
    allow_cross_spread: bool = False
    min_order_value_usd: float = 10.0
    venue_health_threshold: float = 0.7

class ExtremeEventProtectionSettings(BaseModel):
    """Sprint 65 - Comprehensive extreme event protection settings."""
    enabled: bool = True
    shock_detection: ShockDetectionSettings = ShockDetectionSettings()
    circuit_policy: CircuitBreakerPolicySettings = CircuitBreakerPolicySettings()
    alerts: ExtremeEventAlertsSettings = ExtremeEventAlertsSettings()
    safe_exit: SafeExitSettings = SafeExitSettings()

class LiveSettings(BaseModel):
    enabled: bool = False
    dry_run: bool = True
    exchange: str = "binance_usdm"
    symbols: List[str] | None = None
    timeframes: List[str] | None = None
    profiles_root: Optional[str] = None
    hot_reload_profiles: bool = True
    rate_limits: Dict[str, int] = Field(default_factory=lambda: {"orders_per_sec": 8, "cancels_per_sec": 8})
    retries: LiveRetrySettings = LiveRetrySettings()
    circuit_breakers: LiveCircuitBreakers = LiveCircuitBreakers()
    latency: LiveLatencySettings = LiveLatencySettings()
    queues: LiveQueueSettings = LiveQueueSettings()
    # Simulator tuning (dry-run realism)
    simulator: Dict[str, Any] = Field(default_factory=lambda: {
        "partial_fill_prob": 0.2,
        "reject_prob": 0.02,
        "slippage_bps_min": -1.0,
        "slippage_bps_max": 1.5,
        "latency_jitter_ms_max": 25,
    })
    # Metrics / exporter settings
    metrics: Dict[str, Any] = Field(default_factory=lambda: {
        "exporter": "none",  # none|csv (future: prometheus)
    "exporter": "none",  # none|csv|http
        "csv_path": "live_metrics.csv",
        "interval_sec": 10,
    "http_port": 8765,
    "json_log": False,
    })
    # Health / heartbeat settings
    health: Dict[str, Any] = Field(default_factory=lambda: {"heartbeat_interval_sec": 30})
    # Control directory (drop flag files: pause.flag, resume.flag, kill.flag)
    control: Dict[str, Any] = Field(default_factory=lambda: {"control_dir": "live_controls"})


# --- Sprint 40 Sentiment & Social Layer Settings ---
class SentimentSourceToggle(BaseModel):
    enabled: bool = True
    refresh_sec: int = Field(900, gt=0)
    max_items: int = Field(200, gt=0)

class SentimentExtremePolicy(BaseModel):
    z_threshold: float = Field(2.0, gt=0)
    quantile_low: float = Field(0.05, ge=0, le=0.5)
    quantile_high: float = Field(0.95, ge=0.5, le=1.0)
    cooloff_bars: int = Field(10, ge=0)

class SentimentWeightsSettings(BaseModel):
    influencer_weight: float = Field(0.6, ge=0, le=1)
    crowd_weight: float = Field(0.4, ge=0, le=1)
    engagement_weight: float = Field(0.5, ge=0, le=2)
    text_polarity_weight: float = Field(1.0, ge=0, le=2)

class SentimentWindowsSettings(BaseModel):
    short_minutes: int = Field(60, gt=0)
    medium_hours: int = Field(12, gt=0)
    z_lookback: int = Field(240, gt=10)

class SentimentSettings(BaseModel):
    """Settings for the unified sentiment & social signal pipeline (Sprint 40)."""
    enabled: bool = True
    symbols: List[str] | None = None  # default: inherit runtime.symbols if None
    influencers: Dict[str, float] = Field(default_factory=dict, description="handle -> weight map")
    subreddit_list: List[str] = Field(default_factory=lambda: ["CryptoCurrency", "Bitcoin"])
    symbol_keyword_map: Dict[str, List[str]] = Field(default_factory=dict)
    windows: SentimentWindowsSettings = SentimentWindowsSettings()
    weights: SentimentWeightsSettings = SentimentWeightsSettings()
    extremes: SentimentExtremePolicy = SentimentExtremePolicy()
    # Source toggles (all free/no-key by default)
    sources: Dict[str, SentimentSourceToggle] = Field(default_factory=lambda: {
        "twitter": SentimentSourceToggle(refresh_sec=900, max_items=150),
        "reddit": SentimentSourceToggle(refresh_sec=900, max_items=150),
        "discord": SentimentSourceToggle(enabled=False, refresh_sec=1800, max_items=200),
        "fear_greed": SentimentSourceToggle(refresh_sec=3600, max_items=10),
        "funding": SentimentSourceToggle(refresh_sec=900, max_items=50),
        "trends": SentimentSourceToggle(enabled=False, refresh_sec=3600, max_items=50),
        "news": SentimentSourceToggle(enabled=False, refresh_sec=900, max_items=200),
    })
    cache_dir: str = Field(".cache/sentiment", description="Disk cache root for raw & processed sentiment data")
    local_only: bool = False
    transformer_model: Optional[str] = Field(None, description="Optional local HF model name for refined sentiment")
    max_parallel_requests: int = Field(4, gt=0, le=16)
    request_backoff_sec: float = Field(2.0, ge=0)
    request_backoff_max_sec: float = Field(30.0, ge=0)
    veto_extremes: bool = True
    size_dampen_extremes: bool = True
    size_dampen_factor: float = Field(0.5, ge=0, le=1)
    telemetry: Dict[str, Any] = Field(default_factory=lambda: {"emit_metrics": True, "export_path": "sentiment_metrics.csv", "interval_sec": 300})
    # Future placeholder for correlation sanity checks
    diagnostics: Dict[str, Any] = Field(default_factory=lambda: {"rolling_return_corr_bars": 500})


# --- Sprint 23 Venues (multi-venue abstraction) ---
class VenueFees(BaseModel):
    taker: float | None = None
    maker: float | None = None

class VenuesHealthSettings(BaseModel):
    red_threshold: float = 0.35
    yellow_threshold: float = 0.65
    cooloff_sec: int = 30
    staleness_ms_max: int = 2500

class VenueRateLimitSettings(BaseModel):
    rest_rps: float = 8.0
    ws_max_streams: int = 20

class VenuesSettings(BaseModel):
    primary_order: List[str] = Field(default_factory=lambda: ["binance_usdm", "bybit_perp"])
    data_order: List[str] = Field(default_factory=lambda: ["binance_usdm", "bybit_perp"])
    symbol_map: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    health: VenuesHealthSettings = VenuesHealthSettings()
    ratelimits: Dict[str, VenueRateLimitSettings] = Field(default_factory=dict)
    fees: Dict[str, VenueFees] = Field(default_factory=dict)
    prefer_lower_fee_on_tie: bool = True


# --- Sprint 24 Execution Layer Settings (smart limits, brackets, algos) ---
class ExecutionBreakEvenSettings(BaseModel):
    enabled: bool = True
    be_trigger_atr: float = 1.2
    be_lock_ticks: int = 2

class ExecutionTrailingSettings(BaseModel):
    enabled: bool = True
    arm_atr: float = 2.0
    trail_atr_mult: float = 1.0

class ExecutionBracketsSettings(BaseModel):
    enabled: bool = True
    stop_atr_mult: float = 1.4
    tp_atr_mults: List[float] = Field(default_factory=lambda: [1.8, 2.6, 3.5])
    tp_scales: List[float] = Field(default_factory=lambda: [0.5, 0.3, 0.2])
    break_even: ExecutionBreakEvenSettings = ExecutionBreakEvenSettings()
    trailing: ExecutionTrailingSettings = ExecutionTrailingSettings()

class ExecutionTWAPSettings(BaseModel):
    duration_s: int = 120
    slices: int = 6

class ExecutionIcebergSettings(BaseModel):
    clip_usd: float = 3000.0
    refresh_ms: int = 800
    randomize: bool = True

class ExecutionPOVSettings(BaseModel):
    enabled: bool = False
    pct: float = 0.12

class ExecutionAlgosSettings(BaseModel):
    threshold_usd: float = 20000.0
    twap: ExecutionTWAPSettings = ExecutionTWAPSettings()
    iceberg: ExecutionIcebergSettings = ExecutionIcebergSettings()
    pov: ExecutionPOVSettings = ExecutionPOVSettings()

class ExecutionMicrostructureSettings(BaseModel):
    enabled: bool = False
    expected_top_liq: float = 5000.0  # proxy liquidity at top of book
    taker_deadline_bars: int = 3      # convert rest to taker after this many bars
    queue_bars_penalty: int = 0       # delay before first maker fill probability applies
    fill_prob_min: float = 0.05       # floor probability
    fill_prob_max: float = 0.95       # cap probability

class ExecutionSettings(BaseModel):
    maker_first: bool = True
    k1_ticks: int = 1
    taker_fallback_ms: int = 1200
    taker_offset_ticks: int = 1
    max_spread_pct: float = 0.06
    max_chase_bps: int = 8
    atr_pct_limit: float = 0.97
    max_slip_bps: int = 12
    price_anchor: Literal["mid", "vwap", "close"] = "mid"
    brackets: ExecutionBracketsSettings = ExecutionBracketsSettings()
    algos: ExecutionAlgosSettings = ExecutionAlgosSettings()
    microstructure: ExecutionMicrostructureSettings = ExecutionMicrostructureSettings()
    cancel_if_flip: bool = True
    cancel_stale_ms: int = 30_000


# --- Sprint 28 Event Risk Settings ---
class EventActionConfig(BaseModel):
    mode: Literal['VETO','DAMPEN','NONE'] = 'NONE'
    size_mult: float | None = None
    widen_stop_mult: float | None = None
    maker_only: bool | None = None

class EventRiskSettings(BaseModel):
    enabled: bool = False
    pre_window_minutes: Dict[str, int] = Field(default_factory=lambda: {"HIGH":90,"MED":45,"LOW":15})
    post_window_minutes: Dict[str, int] = Field(default_factory=lambda: {"HIGH":90,"MED":30,"LOW":10})
    actions: Dict[str, EventActionConfig] = Field(default_factory=lambda: {
        'HIGH': EventActionConfig(mode='VETO'),
        'MED': EventActionConfig(mode='DAMPEN', size_mult=0.5, widen_stop_mult=1.25, maker_only=True),
        'LOW': EventActionConfig(mode='DAMPEN', size_mult=0.75),
    })
    symbol_overrides: Dict[str, Dict[str, Dict[str, EventActionConfig]]] = Field(default_factory=dict)
    close_existing: Dict[str, bool] = Field(default_factory=lambda: {'HIGH': False})
    cooldown_minutes_after_veto: int = 20
    missing_feed_policy: Literal['SAFE','OPEN','OFF'] = 'SAFE'
    providers: Dict[str, Dict[str, bool]] = Field(default_factory=lambda: {
        'econ_calendar': {'enabled': True},
        'crypto_incidents': {'enabled': True},
    })


# --- Sprint 42 Cross-Asset / Macro Fusion Settings ---
class CrossAssetCorrelationWindow(BaseModel):
    label: str
    bars: int = Field(gt=1)

class CrossAssetTickerGroup(BaseModel):
    # Logical tickers (Yahoo Finance style) and local aliases
    equities: List[str] = Field(default_factory=lambda: ["SPY", "QQQ"])
    fx: List[str] = Field(default_factory=lambda: ["DX-Y.NYB"])  # DXY index
    rates: List[str] = Field(default_factory=lambda: ["^TNX"])  # 10Y yield (Yahoo symbol)
    commodities: List[str] = Field(default_factory=lambda: ["GC=F", "CL=F"])  # Gold / Crude Oil futures
    volatility: List[str] = Field(default_factory=lambda: ["^VIX"])  # CBOE VIX

class CarryUnwindRule(BaseModel):
    dxy_z_thr: float = 1.0
    us10y_z_thr: float = 1.0
    btc_return_thr: float = -0.01  # -1% over lookback
    oi_drop_pct: float = 0.03      # 3% intra-window OI drop
    confirm_bars: int = 2          # consecutive bars to confirm

class MacroRegimeWeights(BaseModel):
    equities_weight: float = 0.35
    volatility_weight: float = 0.25
    fx_weight: float = 0.20
    commodities_weight: float = 0.10
    rates_weight: float = 0.10

class MacroExtremeThresholds(BaseModel):
    corr_extreme_z: float = 2.0
    vix_spike_z: float = 1.5
    dxy_spike_z: float = 1.25
    oil_shock_z: float = 1.5
    gold_flow_z: float = 1.5

class CrossAssetSettings(BaseModel):
    """Sprint 42: Cross-Asset Signal Fusion configuration.

    All components default to disabled to avoid overhead unless explicitly
    turned on in user settings. Data sourcing favors free endpoints (yfinance
    scraping) and optional Deribit public API for implied vol.
    """
    enabled: bool = False
    primary_symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"], description="Crypto symbols to enrich.")
    tickers: CrossAssetTickerGroup = CrossAssetTickerGroup()
    correlation_windows: List[CrossAssetCorrelationWindow] = Field(default_factory=lambda: [
        CrossAssetCorrelationWindow(label="30m", bars=30),
        CrossAssetCorrelationWindow(label="4h", bars=48),
        CrossAssetCorrelationWindow(label="1d", bars=288),
        CrossAssetCorrelationWindow(label="1w", bars=2016),
    ])
    cache_dir: str = ".cache/cross_asset"
    refresh_min: int = Field(5, gt=0)
    realized_vol_window: int = Field(48, gt=5)
    btc_vix_lookback: int = Field(96, gt=10)
    use_deribit_iv: bool = False
    deribit_iv_weight: float = Field(0.5, ge=0, le=1.0)
    deribit_underlyings: List[str] = Field(default_factory=lambda: ["BTC-ETH"])  # placeholder composite toggle
    carry_unwind: CarryUnwindRule = CarryUnwindRule()
    regime_weights: MacroRegimeWeights = MacroRegimeWeights()
    extreme: MacroExtremeThresholds = MacroExtremeThresholds()
    zscore_window: int = Field(200, gt=20)
    # Optional sources toggles
    enable_trends: bool = False  # Google Trends panic proxy
    enable_macro_calendar: bool = False
    enable_news_rss: bool = False
    enable_trading_hours_filter: bool = True
    # Telemetry / diagnostics
    diagnostics: Dict[str, Any] = Field(default_factory=lambda: {
        "emit": False,
        "export_path": "macro_features.csv",  # CSV file or directory if format=parquet
        "append": True,                        # CSV only
        "format": "csv",                      # csv|parquet
        "batch_size": 250,                     # rows before forced flush
        "flush_interval_sec": 60,              # max seconds between flushes
        "async": True                          # flush in background thread
    })
    # Risk-off gating thresholds (applied in ensemble layer)
    risk_off_veto_prob: float = Field(0.72, ge=0, le=1.0)
    risk_off_dampen_prob: float = Field(0.55, ge=0, le=1.0)
    risk_off_conf_mult: float = Field(0.6, ge=0, le=1.0, description="Confidence multiplier when dampening in moderate risk_off regime")


class Settings(BaseModel):
    """The root Pydantic model for the entire configuration."""
    data_sources: Dict[str, DataSourceSettings]
    runtime: RuntimeSettings
    features: FeatureSettings
    derivatives: DerivativesSettings
    regime: RegimeSettings
    weights_profiles: WeightsProfilesSettings
    filters: FiltersSettings
    engine: EngineSettings
    ensemble: EnsembleSettings
    correlation: CorrelationSettings
    portfolio: PortfolioSettings
    portfolio_hedge: Optional[PortfolioHedgeSettings] = None
    brakes: BrakesSettings
    sizing: SizingSettings
    funding_rate_provider: FundingProviderSettings
    transport: TransportSettings
    live: Optional[LiveSettings] = None
    backtest: Optional[BacktestSettings] = None
    walkforward: Optional[WalkforwardSettings] = None
    # Sprint 65 - Extreme Event Protection
    extreme_event_protection: Optional[ExtremeEventProtectionSettings] = None
    calibration: Optional[CalibrationSettings] = None
    reports: Optional[ReportsSettings] = None
    logging: Optional[LoggingSettings] = None
    venues: Optional[VenuesSettings] = None
    execution: Optional[ExecutionSettings] = None  # Sprint 24 execution layer
    event_risk: Optional[EventRiskSettings] = None  # Sprint 28 event gating
    # Sprint 31 Meta-Scorer (kept as loose dict to allow rapid iteration without strict schema churn)
    meta_scorer: Optional[Dict[str, Any]] = None
    # Sprint 39 Data Quality (optional block – loose schema to avoid regressions while iterating)
    data_quality: Optional[Dict[str, Any]] = None
    # Sprint 40 Sentiment & Social Layer
    sentiment: Optional[SentimentSettings] = None
    # Sprint 42 Cross-Asset / Macro Fusion
    cross_asset: Optional[CrossAssetSettings] = None

# --- Helper Functions ---

def _load_config_from_yaml(path: Path) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    if not path.is_file():
        raise ConfigError(f"Configuration file not found at: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML file at {path}: {e}") from e

def _get_env_overrides(prefix: str = "ULTRA_SIGNALS") -> Dict[str, Any]:
    """
    Parses environment variables and converts them into a nested dict.
    e.g., ULTRA_SIGNALS_TRANSPORT__TELEGRAM__BOT_TOKEN becomes
    {'transport': {'telegram': {'bot_token': '...'}}}
    """
    overrides: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and split into parts
            parts = key.removeprefix(prefix).strip("_").lower().split("__")

            # Traverse the dictionary to place the value
            # Explicitly handle bot_token and chat_id as strings
            if 'bot_token' in parts or 'chat_id' in parts:
                parsed_value = value
            # Attempt to parse other values as JSON (for lists, dicts, booleans, numbers)
            elif (value.startswith('[') and value.endswith(']')) or \
                 (value.startswith('{') and value.endswith('}')) or \
                 value.lower() in ['true', 'false', 'null'] or \
                 value.replace('.', '', 1).isdigit():  # Check if it's a number
                try:
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, AttributeError):
                    parsed_value = value
            else:
                parsed_value = value

            d = overrides
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = parsed_value
    return overrides

def _merge_configs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges the override dict into the base dict.
    Overwrites values, dictionaries, and lists.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = _merge_configs(base[key], value)
        else:
            base[key] = value
    return base

# --- Public API ---

def load_settings(path: str = "settings.yaml") -> Settings:
    """
    Loads, validates, and returns the application settings.

    This is the main entry point for configuration. It performs the following steps:
    1. Loads the base configuration from the specified YAML file.
    2. Scans environment variables for overrides (prefixed with "ULTRA_SIGNALS_").
    3. Merges the environment overrides into the base configuration.
    4. Validates the final configuration against the `Settings` Pydantic model.

    Args:
        path: The path to the YAML configuration file.

    Returns:
        A validated and immutable `Settings` object.

    Raises:
        ConfigError: If the file is not found, cannot be parsed, or if
                     validation fails.
    """
    logger.info(f"Loading settings from '{path}'...")

    # 1. Load base config from YAML
    yaml_config = _load_config_from_yaml(Path(path))
    if not yaml_config:
        raise ConfigError(f"YAML file '{path}' is empty or invalid.")

    # 2. Get environment variable overrides
    env_overrides = _get_env_overrides()

    # 3. Merge configs
    final_config = _merge_configs(yaml_config, env_overrides)

    # 4. Validate with Pydantic
    try:
        settings = Settings.model_validate(final_config)
        logger.success("Settings loaded and validated successfully.")
        return settings
    except ValidationError as e:
        error_details = e.errors()
        error_msg = f"Configuration validation failed with {len(error_details)} error(s):\n"
        for error in error_details:
            loc = " -> ".join(map(str, error['loc'])) if error['loc'] else "root"
            error_msg += f"  - Location: {loc}\n    Message: {error['msg']}\n"

        logger.error(error_msg)
        raise ConfigError("Failed to validate settings.") from e

if __name__ == '__main__':
    # Example of how to use the loader
    # This block will run if you execute `python -m core.config`

    # For testing, you can create a dummy .env or set variables
    # For example:
    # export ULTRA_SIGNALS_TRANSPORT__TELEGRAM__BOT_TOKEN="my_test_token"
    # export ULTRA_SIGNALS_RUNTIME__SYMBOLS='["XRPUSDT", "DOGEUSDT"]' # Note JSON format for lists

    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists

    try:
        # Assuming settings.yaml is in the project root
        settings = load_settings()

        # Pretty print the loaded settings
        import json
        print(json.dumps(settings.model_dump(), indent=2))

        # Access settings like an object
        print(f"\nRunning with symbols: {settings.runtime.symbols}")
        print(f"Telegram Dry Run: {settings.transport.dry_run}")
        if settings.transport.telegram.bot_token:
            print("Telegram token found.")
        else:
            print("Telegram token not found (as expected if not in .env).")

    except ConfigError as e:
        print(f"\nCaught a configuration error: {e}")
