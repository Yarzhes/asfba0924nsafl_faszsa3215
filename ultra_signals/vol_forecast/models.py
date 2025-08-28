from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json

try:
    from arch import arch_model
except Exception:  # arch may be optional in dev env
    arch_model = None

from . import persistence


class VolModelManager:
    """Light wrapper to fit GARCH-family models and provide sigma forecasts and persistence.

    New features:
    - save/load models via joblib
    - update registry with last_refit_ts
    """

    def __init__(self, model_type: str = "garch"):
        self.model_type = model_type.lower()
        self.model = None
        self.res = None
        self.last_refit_ts = None
        self.meta: Dict[str, Any] = {}

    def fit(self, returns: pd.Series, refit_ts: Optional[pd.Timestamp] = None, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> Dict[str, Any]:
        # Minimal checks
        if returns.dropna().shape[0] < 30:
            raise ValueError("not enough data to fit volatility model")

        self.last_refit_ts = pd.Timestamp(refit_ts) if refit_ts is not None else pd.Timestamp.now()

        if arch_model is None:
            # fallback: simple EWMA variance
            lam = 0.94
            ewma_var = returns.pow(2).ewm(alpha=1 - lam).mean().iloc[-1]
            self.res = {"ewma_var": ewma_var}
            self.meta = {"model_id": "ewma_fallback", "ewma_var": float(ewma_var)}
            # persist metadata
            if symbol:
                persistence.update_registry(symbol, timeframe or "default", self.meta, self.last_refit_ts)
            return self.meta

        p = 1
        q = 1

        if self.model_type in ("garch", "garch11"):
            am = arch_model(returns * 100.0, vol="Garch", p=p, q=q, dist="normal")
        elif self.model_type in ("egarch",):
            am = arch_model(returns * 100.0, vol="EGARCH", p=p, q=q, dist="normal")
        elif self.model_type in ("tgarch", "gjr"):
            am = arch_model(returns * 100.0, vol="GARCH", p=p, q=q, o=1, power=2.0, dist="normal")
        else:
            am = arch_model(returns * 100.0, vol="Garch", p=p, q=q, dist="normal")

        try:
            self.res = am.fit(disp="off")
            self.model = am
            self.meta = {"model_id": f"{self.model_type}", "params": self.res.params.to_dict()}
            # persist model and registry
            if symbol:
                persistence.save_model(self, symbol, timeframe or "default")
                persistence.update_registry(symbol, timeframe or "default", self.meta, self.last_refit_ts)
            return self.meta
        except Exception as e:
            # fallback to EWMA
            lam = 0.94
            ewma_var = returns.pow(2).ewm(alpha=1 - lam).mean().iloc[-1]
            self.res = {"ewma_var": ewma_var}
            self.meta = {"model_id": "ewma_fallback", "error": str(e), "ewma_var": float(ewma_var)}
            if symbol:
                persistence.update_registry(symbol, timeframe or "default", self.meta, self.last_refit_ts)
            return self.meta

    def forecast_sigma(self, horizon: int = 1) -> float:
        """Return per-bar sigma (not annualized) for next `horizon` bars."""
        if self.res is None:
            raise RuntimeError("model not fitted")

        if isinstance(self.res, dict) and "ewma_var" in self.res:
            var = float(self.res["ewma_var"])
            # sigma per-bar as sqrt(var); note returns were scaled by 100 in arch fallback
            return float(np.sqrt(var)) / 100.0

        # use arch forecast
        try:
            f = self.res.forecast(horizon=horizon, reindex=False)
            # conditional variance for last available obs -> horizon-1 index
            var = f.variance.values[-1, -1]
            sigma = float(np.sqrt(var)) / 100.0
            return sigma
        except Exception:
            # last-resort fallback
            try:
                return float(np.sqrt(self.res.conditional_volatility.iloc[-1])) / 100.0
            except Exception:
                raise

    @staticmethod
    def load(symbol: str, timeframe: str = "default") -> Optional["VolModelManager"]:
        return persistence.load_model(symbol, timeframe)

