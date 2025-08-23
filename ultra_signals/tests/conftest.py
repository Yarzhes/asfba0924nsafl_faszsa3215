"""
Pytest Fixtures for the Ultra-Signals Test Suite

This file defines shared fixtures that can be used across multiple test files.
Fixtures are a powerful feature of pytest that allow for setting up a
well-defined, consistent context for tests, such as loading configuration
or creating mock objects.

https://docs.pytest.org/en/latest/how-to/fixtures.html
"""
import numpy as np
import pandas as pd
import pytest
from ultra_signals.core.config import Settings  # Adjust this import based on your actual structure

@pytest.fixture(scope="session")
def settings_fixture() -> Settings:
    """
    A pytest fixture that loads a default, validated Settings object.
    This can be used in any test that requires access to the application config.
    The `scope="session"` means this is created only once per test run.
    """
    # Create a minimal but valid configuration for testing purposes.
    # This avoids dependency on a physical `settings.yaml` file for unit tests.
    test_config = {
        "data_sources": {
            "binance_usdm": {
                "api_key": "test_key",
                "api_secret": "test_secret"
            }
        },
        "runtime": {
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "timeframes": ["1m", "5m"],
            "primary_timeframe": "5m",
            "reconnect_backoff_ms": 100,
        },
        "features": {
            "warmup_periods": 50,
            "trend": {"ema_short": 10, "ema_medium": 20, "ema_long": 50},
            "momentum": {"rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
            "volatility": {"atr_period": 14, "bbands_period": 20, "bbands_stddev": 2},
            "volume_flow": {"vwap_window": 20, "volume_z_window": 50},
        },
        "engine": {
            "scoring_weights": {"trend": 0.5, "momentum": 0.5, "volatility": 0.0, "orderbook": 0.0, "derivatives": 0.0},
            "thresholds": {"enter": 0.7, "exit": 0.5},
            "risk": {"max_spread_pct": {"default": 0.1}, "avoid_funding_minutes": 5},
        },
        "transport": {
            "telegram": {"enabled": False, "bot_token": None, "chat_id": None},
            "dry_run": True,
        },
        "funding_rate_provider": {
            "refresh_interval_minutes": 15,
        },
        "derivatives": {
            "oi": {"enabled": True},
            "liq_pulse": {"enabled": True, "timeframe_ms": 300000},
        },
        "regime": {"adx_period": 14, "ema_period": 20},
        "weights_profiles": {
            "trend": {
                "trend": 0.6,
                "momentum": 0.4,
                "volatility": 0.0,
                "orderbook": 0.0,
                "derivatives": 0.0,
                "pullback_confluence": 0.0,
                "breakout_confluence": 0.0,
                "flow": 0.0,
            },
            "mean_revert": {
                "trend": 0.2,
                "momentum": 0.3,
                "volatility": 0.0,
                "orderbook": 0.5,
                "derivatives": 0.0,
                "pullback_confluence": 0.0,
                "breakout_confluence": 0.0,
                "flow": 0.0,
            },
            "chop": {
                "trend": 0.1,
                "momentum": 0.1,
                "volatility": 0.0,
                "orderbook": 0.8,
                "derivatives": 0.0,
                "pullback_confluence": 0.0,
                "breakout_confluence": 0.0,
                "flow": 0.0,
            },
        },
        "filters": {
            "avoid_funding_minutes": 5,
        },
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
    return Settings.model_validate(test_config)


@pytest.fixture(scope="session")
def ohlcv_fixture() -> pd.DataFrame:
    """
    Provides a sample OHLCV DataFrame for testing feature calculations.
    This data is deterministic and can be used to check for expected outputs.
    """
    # Generate some synthetic data that is more interesting than random noise.
    # A simple sine wave with some trend and noise.
    periods = 200
    price = (
        100
        + pd.Series(np.sin(np.linspace(0, 10, periods))) * 5
        + pd.Series(np.linspace(0, 20, periods)) # Add a trend component
        + pd.Series(np.random.randn(periods) * 0.5) # Add noise
    )
    
    data = {
        'timestamp': pd.to_datetime(np.arange(periods), unit='m', origin='2023-01-01'),
        'open': price,
        'high': price + 0.5,
        'low': price - 0.5,
        'close': price.shift(-1).ffill(), # Make close slightly different
        'volume': pd.Series(np.random.randint(100, 1000, periods), dtype=float),
    }
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df