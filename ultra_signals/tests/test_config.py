"""
Tests for the configuration loading and validation logic.
"""

import pytest
import yaml
from copy import deepcopy

from ultra_signals.core.config import ConfigError, load_settings, Settings

# A minimal, valid config dictionary for creating test YAML files
VALID_CONFIG_DICT = {
    "data_sources": {"binance_usdm": {}},
    "runtime": {
        "symbols": ["BTCUSDT"],
        "timeframes": ["1m", "5m"],
        "primary_timeframe": "5m",
        "reconnect_backoff_ms": 1000,
    },
    "features": {
        "warmup_periods": 50,
        "trend": {"ema_short": 10, "ema_medium": 20, "ema_long": 50},
        "momentum": {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        },
        "volatility": {"atr_period": 14, "bbands_period": 20, "bbands_stddev": 2.0},
        "volume_flow": {"vwap_window": 20, "volume_z_window": 50, "cvd_lookback_seconds": 300, "cvd_slope_period": 20,},
        "orderbook_v2": {
            "depth_levels_N": 10,
            "book_flip_min_delta": 0.15,
            "book_flip_persistence_ticks": 5,
            "slippage_trade_size": 1000,
        }
    },
    "engine": {
        "scoring_weights": {
            "trend": 0.5,
            "momentum": 0.5,
            "volatility": 0.0,
            "orderbook": 0.0,
            "derivatives": 0.0,
        },
        "thresholds": {"enter": 0.7, "exit": 0.5},
        "risk": {
            "max_spread_pct": {"default": 0.1},
            "avoid_funding_minutes": 5,
            "breakout_confirmation": {"enabled": False},
            "mean_reversion_filter": {"enabled": False},
            "cvd_alignment": {"enabled": False},
            "slippage_cap": {"enabled": False},
        },
    },
    "transport": {
        "telegram": {"enabled": False},
        "dry_run": True,
    },
    "funding_rate_provider": {"refresh_interval_minutes": 15},
    "derivatives": {
        "oi": {"enabled": False},
        "liq_pulse": {"enabled": False, "timeframe_ms": 300000},
    },
    "regime": {"adx_period": 14, "ema_period": 20},
    "weights_profiles": {
        "trend": {"trend": 1.0, "momentum": 0.0, "volatility": 0.0, "orderbook": 0.0, "derivatives": 0.0, "pullback_confluence": 0.0, "breakout_confluence": 0.0, "flow": 0.0},
        "mean_revert": {"trend": 0.0, "momentum": 1.0, "volatility": 0.0, "orderbook": 0.0, "derivatives": 0.0, "pullback_confluence": 0.0, "breakout_confluence": 0.0, "flow": 0.0},
        "chop": {"trend": 0.0, "momentum": 0.0, "volatility": 1.0, "orderbook": 0.0, "derivatives": 0.0, "pullback_confluence": 0.0, "breakout_confluence": 0.0, "flow": 0.0},
    },
    "filters": {"avoid_funding_minutes": 5},
    "ensemble": {
        "majority_threshold": 0.6,
        "veto_trend_flip": True,
        "veto_band_pierce": True,
    },
    "correlation": {
        "enabled": True,
        "lookback_periods": 200,
        "cluster_threshold": 0.7,
        "hysteresis": 0.1,
        "refresh_interval_bars": 50,
    },
    "portfolio": {
        "max_exposure_per_symbol": 1000.0,
        "max_exposure_per_cluster": 3000.0,
        "max_net_exposure": 5000.0,
        "max_margin_pct": 50.0,
        "max_total_positions": 10,
    },
    "brakes": {
        "min_spacing_sec_cluster": 300,
        "daily_loss_soft_limit_pct": 2.0,
        "daily_loss_hard_limit_pct": 4.0,
        "streak_cooldown_trades": 3,
        "streak_cooldown_hours": 12,
    },
    "sizing": {
        "vol_risk_scale_pct": 0.5,
    },
}


def test_load_settings_success(tmp_path):
    """
    Tests that a valid YAML file is loaded correctly into a Settings object.
    """
    config_file = tmp_path / "settings.yaml"
    with open(config_file, "w") as f:
        yaml.dump(VALID_CONFIG_DICT, f)

    settings = load_settings(path=str(config_file))

    assert isinstance(settings, Settings)
    assert settings.runtime.symbols == ["BTCUSDT"]
    assert settings.features.trend.ema_long == 50
    assert settings.transport.dry_run is True


def test_load_settings_file_not_found():
    """
    Tests that a ConfigError is raised if the settings file does not exist.
    """
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_settings(path="non_existent_file.yaml")


def test_config_validation_error_missing_key(tmp_path):
    """
    Tests that a ConfigError is raised if a required key is missing.
    """
    invalid_config = deepcopy(VALID_CONFIG_DICT)
    del invalid_config["runtime"]  # 'runtime' is a required section

    config_file = tmp_path / "invalid.yaml"
    with open(config_file, "w") as f:
        yaml.dump(invalid_config, f)

    with pytest.raises(ConfigError, match="Failed to validate settings"):
        load_settings(path=str(config_file))


def test_config_validation_error_wrong_type(tmp_path):
    """
    Tests that a ConfigError is raised if a value has the wrong type.
    """
    invalid_config = deepcopy(VALID_CONFIG_DICT)
    # 'symbols' should be a list, not a string
    invalid_config["runtime"]["symbols"] = "not-a-list"

    config_file = tmp_path / "invalid.yaml"
    with open(config_file, "w") as f:
        yaml.dump(invalid_config, f)

    with pytest.raises(ConfigError, match="Failed to validate settings"):
        load_settings(path=str(config_file))


def test_primary_timeframe_validation(tmp_path):
    """
    Tests the custom validator for `primary_timeframe`.
    """
    invalid_config = deepcopy(VALID_CONFIG_DICT)
    # '10m' is not in the 'timeframes' list
    invalid_config["runtime"]["symbols"] = ["BTCUSDT"] # Ensure symbols is a list
    invalid_config["runtime"]["primary_timeframe"] = "10m"

    config_file = tmp_path / "invalid.yaml"
    with open(config_file, "w") as f:
        yaml.dump(invalid_config, f)

    with pytest.raises(ConfigError, match="Failed to validate settings"):
        load_settings(path=str(config_file))


def test_env_override(tmp_path, monkeypatch):
    """
    Tests that environment variables correctly override YAML settings.
    """
    config_file = tmp_path / "settings.yaml"
    with open(config_file, "w") as f:
        yaml.dump(VALID_CONFIG_DICT, f)

    import json
    # Set environment variables to override specific settings
    monkeypatch.setenv("ULTRA_SIGNALS_RUNTIME__SYMBOLS", json.dumps(["ETHUSDT", "SOLUSDT"]))
    monkeypatch.setenv("ULTRA_SIGNALS_TRANSPORT__TELEGRAM__BOT_TOKEN", "my_env_token")
    monkeypatch.setenv("ULTRA_SIGNALS_TRANSPORT__DRY_RUN", "false") # Note: string 'false' becomes bool False

    # This time, we don't try to override with an invalid timeframe
    settings = load_settings(path=str(config_file))

    assert settings.runtime.symbols == ["ETHUSDT", "SOLUSDT"]
    assert settings.transport.telegram.bot_token == "my_env_token"
    assert settings.transport.dry_run is False
    # The primary timeframe should remain the default from the file
    assert settings.runtime.primary_timeframe == "5m"