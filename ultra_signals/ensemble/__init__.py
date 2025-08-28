"""Ensemble package: scaffolding for multi-model stacking, calibration, and sizing.

This module contains lightweight interfaces used by higher-level orchestration.
Implementations (XGBoost/LightGBM/LSTM/Transformer/RL) belong in dedicated
modules and will be registered with the pipeline via the ModelRegistry.
"""

from .pipeline import EnsemblePipeline
from .stacker import Stacker

__all__ = ["EnsemblePipeline", "Stacker"]
