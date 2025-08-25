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

class FeatureSettings(BaseModel):
    """Container for all feature computation parameters."""
    warmup_periods: int = Field(gt=1)
    trend: TrendFeatureSettings
    momentum: MomentumFeatureSettings
    volatility: VolatilityFeatureSettings
    volume_flow: VolumeFlowFeatureSettings

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

class RegimeSettings(BaseModel):
    """Parameters for the market regime classification engine."""
    hysteresis_bars: int = Field(3, ge=0)
    adx_min_trend: int = Field(15, ge=0, le=100)
    atr_percentile_windows: int = Field(200, gt=1)
    variance_ratio_window: int = Field(20, gt=1)

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
    daily_loss_limit_pct: float = 0.06
    max_consecutive_losses: int = 4
    order_error_burst: LiveOrderErrorBurst = LiveOrderErrorBurst()
    data_staleness_ms: int = 2500

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
        "csv_path": "live_metrics.csv",
        "interval_sec": 10,
    })
    # Health / heartbeat settings
    health: Dict[str, Any] = Field(default_factory=lambda: {"heartbeat_interval_sec": 30})
    # Control directory (drop flag files: pause.flag, resume.flag, kill.flag)
    control: Dict[str, Any] = Field(default_factory=lambda: {"control_dir": "live_controls"})

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
    brakes: BrakesSettings
    sizing: SizingSettings
    funding_rate_provider: FundingProviderSettings
    transport: TransportSettings
    live: Optional[LiveSettings] = None
    backtest: Optional[BacktestSettings] = None
    walkforward: Optional[WalkforwardSettings] = None
    calibration: Optional[CalibrationSettings] = None
    reports: Optional[ReportsSettings] = None
    logging: Optional[LoggingSettings] = None

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
