"""Volatility forecasting bundle (GARCH/TGARCH/EGARCH + realized vol fallbacks).

Lightweight, pluggable wrapper exposing a simple API:
 - fit/update models from OHLCV DataFrame
 - forecast per-bar sigma for multiple horizons
 - compute VaR and regime tags

This is an initial implementation matching Sprint 52 acceptance criteria.
"""

from .models import VolModelManager
from .pipeline import prepare_returns, forecast_vols

__all__ = ["VolModelManager", "prepare_returns", "forecast_vols"]
