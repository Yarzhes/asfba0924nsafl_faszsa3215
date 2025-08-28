"""Online Kyle's lambda estimator (rolling robust regression).

Provides a small, dependency-light estimator that fits ΔP = λ * ΔQ using a
rolling window and robust loss (Huber when scikit-learn is available, else OLS).

Exports KyleLambdaEstimator which maintains recent (dq, dp) pairs, computes
lambda, lambda_bps_per_1k, z-score vs rolling median, and basic confidence (R^2).
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional
import math
import statistics

import numpy as np

try:
    from sklearn.linear_model import HuberRegressor
    _HAS_SK = True
except Exception:
    HuberRegressor = None
    _HAS_SK = False


@dataclass
class LambdaSnapshot:
    ts: int
    lambda_est: float
    r2: float
    samples: int
    lambda_bps_per_1k: Optional[float]
    lambda_z: Optional[float]
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None


class KyleLambdaEstimator:
    def __init__(self, window: int = 200, min_samples: int = 30):
        """Create estimator.

        Args:
            window: number of recent samples to keep (dq, dp pairs).
            min_samples: minimum samples before producing an estimate.
        """
        self.window = int(window)
        self.min_samples = int(min_samples)
        self._dq: Deque[float] = deque(maxlen=self.window)
        self._dp: Deque[float] = deque(maxlen=self.window)
        self._prices: Deque[float] = deque(maxlen=self.window)  # recent mid prices for notional conversions
        # historical lambda values for median/mad z-score
        self._lambda_history: Deque[float] = deque(maxlen=1000)

        # last computed
        self.lambda_est: Optional[float] = None
        self.r2: Optional[float] = None
        self.samples: int = 0

    def add_observation(self, dq: float, dp: float, mid_price: Optional[float] = None, ts: int = 0):
        """Add one window observation (signed volume, mid-price change).

        dq: signed volume (units)
        dp: mid-price change (price units)
        mid_price: optional mid price at that time for notional conversions
        ts: timestamp (unused currently, stored in snapshot if needed)
        """
        if dq is None or dp is None:
            return
        self._dq.append(float(dq))
        self._dp.append(float(dp))
        if mid_price is not None:
            self._prices.append(float(mid_price))
        self.samples = len(self._dq)
        if self.samples >= self.min_samples:
            self._fit()

    def _fit(self):
        x = np.array(self._dq).reshape(-1, 1)
        y = np.array(self._dp)
        coef = 0.0
        r2 = 0.0
        try:
            if _HAS_SK:
                # Huber is robust to outliers
                model = HuberRegressor().fit(x, y)
                coef = float(model.coef_[0])
                y_pred = model.predict(x)
            else:
                # fallback to simple OLS with intercept forced zero
                coef = float(np.sum(x.flatten() * y) / max(1e-12, np.sum(x.flatten() * x.flatten())))
                y_pred = coef * x.flatten()
            # R^2
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except Exception:
            coef = 0.0
            r2 = 0.0

        self.lambda_est = coef
        self.r2 = float(r2)
        # record history for z-score
        try:
            self._lambda_history.append(self.lambda_est)
        except Exception:
            pass

    def snapshot(self, ts: int = 0) -> LambdaSnapshot:
        # compute lambda_bps_per_1k using last mid_price if available
        last_price = None
        if len(self._prices) > 0:
            last_price = float(self._prices[-1])
        lbps = None
        if self.lambda_est is not None and last_price and last_price > 0:
            # ΔP for $1k notional = lambda * (1000 / price)
            dp_1k = self.lambda_est * (1000.0 / last_price)
            lbps = (dp_1k / last_price) * 10_000.0

        # z-score vs rolling median/mad
        lz = None
        if self._lambda_history and len(self._lambda_history) >= max(5, self.min_samples):
            med = statistics.median(self._lambda_history)
            # robust scale: median absolute deviation
            mad = statistics.median([abs(v - med) for v in self._lambda_history]) or 1e-12
            lz = (self.lambda_est - med) / (1.4826 * mad)

        return LambdaSnapshot(ts=ts, lambda_est=float(self.lambda_est or 0.0), r2=float(self.r2 or 0.0), samples=self.samples, lambda_bps_per_1k=lbps, lambda_z=lz)


__all__ = ['KyleLambdaEstimator', 'LambdaSnapshot']
