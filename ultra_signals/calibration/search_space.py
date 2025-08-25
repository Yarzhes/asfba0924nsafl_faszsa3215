"""Sprint 19: Search Space Definitions

Defines parameter ranges for Bayesian optimization / random search.
The config-driven approach lets cal_config.yaml specify bounds. This module
simply converts YAML dict spec -> helper functions for Optuna trial sampling.
"""
from __future__ import annotations
from typing import Dict, Any, Callable

NUMERIC_KEYS = {"low", "high", "step"}

class SearchSpace:
    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec or {}

    def sample(self, trial) -> Dict[str, Any]:  # Optuna trial interface (has suggest_* methods)
        params = {}
        for group, gspec in self.spec.items():
            if not isinstance(gspec, dict):
                continue
            for key, bounds in gspec.items():
                full_key = f"{group}.{key}"
                if isinstance(bounds, dict) and NUMERIC_KEYS & set(bounds.keys()):
                    low = bounds.get("low")
                    high = bounds.get("high")
                    step = bounds.get("step")
                    if isinstance(low, int) and isinstance(high, int) and (step is not None):
                        params[full_key] = trial.suggest_int(full_key, low, high, step=step)
                    elif isinstance(low, int) and isinstance(high, int):
                        params[full_key] = trial.suggest_int(full_key, low, high)
                    else:
                        params[full_key] = float(trial.suggest_float(full_key, float(low), float(high)))
                else:
                    # treat as categorical
                    if isinstance(bounds, (list, tuple)):
                        params[full_key] = trial.suggest_categorical(full_key, list(bounds))
                    else:
                        # single fixed value
                        params[full_key] = bounds
        return params

    def validate_within_bounds(self, params: Dict[str, Any]) -> bool:
        for group, gspec in self.spec.items():
            for key, bounds in gspec.items():
                full_key = f"{group}.{key}"
                if full_key not in params:
                    continue
                val = params[full_key]
                if isinstance(bounds, dict) and "low" in bounds and "high" in bounds:
                    lo, hi = bounds["low"], bounds["high"]
                    if val < lo or val > hi:
                        return False
        return True

__all__ = ["SearchSpace"]
