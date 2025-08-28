"""Online Kyle estimator with time-window aggregation and EW regression.

Provides:
- TimeWindowAggregator: accumulate trades/mid updates and produce (dq, dp) over a sliding seconds window.
- EWKyleEstimator: exponential-weighted online estimator computing lambda via EW covariances.

This is intentionally dependency-light and designed for integration into live feature pipelines.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple, Optional
import math

from .kyle_lambda import LambdaSnapshot


@dataclass
class Tick:
    ts_ms: int
    mid: float
    signed_size: float


class TimeWindowAggregator:
    """Accumulates ticks and emits (dq, dp, mid_at_end) over a sliding window (seconds).

    Usage: call add_tick(ts_ms, mid, signed_size) for each trade or quote update.
    Call window_sample(now_ms) to get aggregated signed flow and mid-price change over the window.
    """
    def __init__(self, window_s: float = 5.0):
        self.window_ms = int(max(1, window_s * 1000))
        self._buf: Deque[Tick] = deque()

    def add_tick(self, ts_ms: int, mid: float, signed_size: float):
        self._buf.append(Tick(ts_ms=int(ts_ms), mid=float(mid), signed_size=float(signed_size)))
        self._expire(ts_ms)

    def _expire(self, now_ms: int):
        cutoff = now_ms - self.window_ms
        while self._buf and self._buf[0].ts_ms < cutoff:
            self._buf.popleft()

    def window_sample(self, now_ms: Optional[int] = None) -> Tuple[float, float, Optional[float]]:
        """Return (dq, dp, last_mid) computed over the window ending at now_ms (or last tick ts if None).

        dq = sum of signed_size in the window
        dp = mid_last - mid_first (None if insufficient samples)
        """
        if not self._buf:
            return 0.0, 0.0, None
        if now_ms is None:
            now_ms = self._buf[-1].ts_ms
        self._expire(now_ms)
        if not self._buf:
            return 0.0, 0.0, None
        dq = sum(t.signed_size for t in self._buf)
        if len(self._buf) >= 2:
            dp = float(self._buf[-1].mid - self._buf[0].mid)
        else:
            dp = 0.0
        return float(dq), float(dp), float(self._buf[-1].mid)


class EWKyleEstimator:
    """Exponential-weighted online estimator.

    Maintains EW moments to estimate lambda = cov(dq, dp) / var(dq).
    alpha: forgetting factor in (0,1) — higher alpha means faster forgetting (per-sample weight = alpha).
    """
    def __init__(self, alpha: float = 0.02):
        if not (0.0 < alpha < 1.0):
            raise ValueError('alpha must be in (0,1)')
        self.alpha = float(alpha)

        # EW means and second moments
        self.ew_x = 0.0
        self.ew_y = 0.0
        self.ew_x2 = 0.0
        self.ew_xy = 0.0
        self.ew_y2 = 0.0

        # initialization flag and effective sample count
        self.initialized = False
        self.n_eff = 0.0

        # results
        self.lambda_est = 0.0
        self.r2 = 0.0

        # For simple CI: track EW of squared residuals (y - lam*x)^2 via plug-in
        self.ew_res2 = 0.0

        # small sample protection
        self._min_var_x = 1e-18

    def add_sample(self, dq: float, dp: float, *, use_notional: bool = False, notional: float = 0.0):
        """Add a (dq, dp) observation. If use_notional=True, then `notional` (signed) is used as x instead of dq.

        dq: signed size (units)
        dp: mid-price change (price units)
        use_notional: if True, use provided notional as the regressor x
        notional: signed notional (price * qty * sign)
        """
        x = float(notional if use_notional and notional is not None else dq)
        y = float(dp)
        if not self.initialized:
            # bootstrap
            self.ew_x = x
            self.ew_y = y
            self.ew_x2 = x * x
            self.ew_xy = x * y
            self.ew_y2 = y * y
            self.initialized = True
            self.n_eff = 1.0
        else:
            a = self.alpha
            self.ew_x = a * x + (1 - a) * self.ew_x
            self.ew_y = a * y + (1 - a) * self.ew_y
            self.ew_x2 = a * (x * x) + (1 - a) * self.ew_x2
            self.ew_xy = a * (x * y) + (1 - a) * self.ew_xy
            self.ew_y2 = a * (y * y) + (1 - a) * self.ew_y2
            self.n_eff = (1 - (1 - a) ** (self.n_eff + 1)) / a if self.n_eff > 0 else 1.0

        # cov and var
        cov_xy = self.ew_xy - (self.ew_x * self.ew_y)
        var_x = max(self._min_var_x, self.ew_x2 - (self.ew_x * self.ew_x))
        var_y = max(1e-18, self.ew_y2 - (self.ew_y * self.ew_y))
        lam = cov_xy / var_x if var_x > 0 else 0.0
        r2 = (cov_xy * cov_xy) / (var_x * var_y) if var_x > 0 and var_y > 0 else 0.0
        self.lambda_est = float(lam)
        self.r2 = float(max(0.0, min(1.0, r2)))
        # update EW residual second moment for simple variance estimate
        # residual = y - lam*x
        res = y - self.lambda_est * x
        # if effectively first sample use direct init, else EW update
        if (self.n_eff or 0.0) <= 1.0:
            self.ew_res2 = res * res
        else:
            a = self.alpha
            self.ew_res2 = a * (res * res) + (1 - a) * self.ew_res2

    def snapshot(self, last_mid: Optional[float] = None, ts: int = 0) -> LambdaSnapshot:
        # lambda_bps_per_1k requires a mid price
        lbps = None
        if last_mid is not None and last_mid > 0:
            dp_1k = self.lambda_est * (1000.0 / last_mid)
            lbps = (dp_1k / last_mid) * 10_000.0
        # sample count approx
        samples = int(self.n_eff) if self.n_eff else 0
        # approximate standard error: se_lambda ~= sqrt( Var(res) / (n_eff * Var(x)) )
        var_x = max(self._min_var_x, self.ew_x2 - (self.ew_x * self.ew_x))
        var_res = max(1e-18, self.ew_res2)
        se = math.sqrt(var_res / max(1.0, (self.n_eff or 1.0)) / var_x)
        ci_low = float(self.lambda_est - 1.96 * se)
        ci_high = float(self.lambda_est + 1.96 * se)
        # z-score like stat (lambda / se)
        lambda_z = float(self.lambda_est / (se + 1e-18)) if se > 0 else None
        return LambdaSnapshot(
            ts=ts,
            lambda_est=float(self.lambda_est or 0.0),
            r2=float(self.r2 or 0.0),
            samples=samples,
            lambda_bps_per_1k=lbps,
            lambda_z=lambda_z,
            ci_low=ci_low,
            ci_high=ci_high,
        )


__all__ = ['TimeWindowAggregator', 'EWKyleEstimator']
