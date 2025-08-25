"""
Feature Engineering Package for Ultra-Signals

This package contains modules for calculating various types of technical
indicators from OHLCV data. Each module focuses on a specific category
of features (e.g., trend, momentum, volatility).

The functions exposed here are the primary interface for the rest of the
application to access feature computations.
"""

# Import the primary computation functions to make them accessible
# directly from the `features` package.
from .momentum import compute_momentum_features
from .trend import compute_trend_features
from .volatility import compute_volatility_features
from .volume_flow import compute_volume_flow_features
from .orderbook import compute_orderbook_features
from .derivatives import compute_derivatives_features
from .funding import compute_funding_features
from .cvd import compute_cvd_features
from .alpha_v2 import compute_alpha_v2_features


# You can define a __all__ to control what `from features import *` does,
# which is good practice for libraries.
__all__ = [
    "compute_trend_features",
    "compute_momentum_features",
    "compute_volatility_features",
    "compute_volume_flow_features",
    "compute_orderbook_features",
    "compute_derivatives_features",
    "compute_funding_features",
    "compute_cvd_features",
    "compute_alpha_v2_features",
]