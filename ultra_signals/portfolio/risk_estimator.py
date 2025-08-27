"""Portfolio Risk Estimator (Sprint 33)

Maintains rolling volatility (ATR-analog via true range) and correlation
matrices for symbols in the primary timeframe. Provides lightweight, cached
accessors for allocator logic.

Design goals:
- Pure python / numpy (pandas optional) minimal overhead (< ~1ms typical)
- Incremental updates per bar (append O(1), occasional matrix recompute)
- Fallback safe defaults when insufficient data (min_bars not met)

Consistency: AdvancedSizer & legacy sizing use ATR; we approximate per-bar
volatility as avg True Range over N lookback bars.

Public API:
    update(symbol, ohlcv_row: dict|pd.Series, ts_epoch: int)
    get_vol(symbol) -> float  # per-bar TR mean (can be annualized externally if needed)
    get_corr(a,b) -> float
    get_cluster(symbol) -> str|None
    active_clusters(open_positions) -> set[str]

Internal representation:
    _history[symbol] = deque[(high, low, close)] length<=lookback+1
    We compute True Range sequence then rolling mean TR as volatility proxy.
    Correlation uses log returns of close.

PSD safety: correlation matrix derived from covariance; tiny negative eigen
values (due to numerical noise) are floored at 1e-9 and matrix reconstructed.

Author: Sprint 33
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple, Optional, Iterable
import math
import numpy as np

try:  # optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


@dataclass
class VolCorrSnapshot:
    vols: Dict[str, float]
    corr: Dict[Tuple[str, str], float]
    ts: int


class RiskEstimator:
    def __init__(self, settings: Dict):
        pr_cfg = (settings.get('portfolio_risk') or {}) if isinstance(settings, dict) else {}
        self.lookback = int(pr_cfg.get('lookback_bars', 288))
        self.min_bars = int(pr_cfg.get('min_bars', max(32, self.lookback//3)))
        self.cluster_map = pr_cfg.get('clusters', {}) or {}
        self.corr_floor_ignore = float(pr_cfg.get('corr_floor', 0.0))  # for external logic
        self._history: Dict[str, Deque[Tuple[float, float, float]]] = {}
        self._closes: Dict[str, Deque[float]] = {}
        self._vol_cache: Dict[str, float] = {}
        self._corr_cache: Dict[Tuple[str, str], float] = {}
        self._last_matrix_ts: int = 0
        self._last_update_ts: int = 0

    # ------------- helpers -------------
    @staticmethod
    def _true_range(prev_close: float, high: float, low: float, close: float) -> float:
        return max(high - low, abs(high - prev_close), abs(low - prev_close))

    def update(self, symbol: str, ohlcv_row: dict, ts_epoch: int) -> None:
        """Append latest bar for symbol. ohlcv_row must have high, low, close.
        Accepts dict or pandas Series. Silently ignores bad inputs."""
        try:
            high = float(ohlcv_row.get('high'))
            low = float(ohlcv_row.get('low'))
            close = float(ohlcv_row.get('close'))
        except Exception:
            return
        if any(math.isnan(x) or math.isinf(x) for x in (high, low, close)):
            return
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self.lookback + 2)
        if symbol not in self._closes:
            self._closes[symbol] = deque(maxlen=self.lookback + 2)
        self._history[symbol].append((high, low, close))
        self._closes[symbol].append(close)
        self._last_update_ts = ts_epoch
        # Invalidate per-symbol vol cache lazily
        self._vol_cache.pop(symbol, None)

    # ------------- volatility -------------
    def _compute_vol(self, symbol: str) -> float:
        dq = self._history.get(symbol)
        if not dq or len(dq) < 2:
            return 0.0
        tr_vals = []
        prev_close = dq[0][2]
        for h, l, c in list(dq)[1:]:
            tr_vals.append(self._true_range(prev_close, h, l, c))
            prev_close = c
        if not tr_vals:
            return 0.0
        # Simple mean TR (can swap to EMA if desired)
        return float(np.mean(tr_vals[-self.lookback:]))

    def get_vol(self, symbol: str) -> float:
        if symbol in self._vol_cache:
            return self._vol_cache[symbol]
        v = self._compute_vol(symbol)
        self._vol_cache[symbol] = v
        return v

    # ------------- correlation -------------
    def _recompute_corr_matrix(self) -> None:
        symbols = [s for s, dq in self._closes.items() if len(dq) >= self.min_bars]
        if len(symbols) < 2:
            self._corr_cache.clear()
            return
        # Build log return matrix
        rets = []
        valid_symbols = []
        for s in symbols:
            closes = np.array(self._closes[s], dtype=float)
            if len(closes) < 2:
                continue
            r = np.diff(np.log(closes))
            if len(r) >= self.lookback:
                r = r[-self.lookback:]
            # align lengths by padding in front
            rets.append(r)
            valid_symbols.append(s)
        if len(rets) < 2:
            self._corr_cache.clear(); return
        # Pad to same length
        max_len = max(len(r) for r in rets)
        mat = []
        for r in rets:
            if len(r) < max_len:
                r = np.concatenate([np.full(max_len - len(r), np.nan), r])
            mat.append(r)
        X = np.array(mat)
        # drop rows with NaN across any symbol
        # (simple: mask columns with NaNs)
        mask = ~np.any(np.isnan(X), axis=0)
        X = X[:, mask]
        if X.shape[1] < 2:
            self._corr_cache.clear(); return
        # covariance
        Xc = X - X.mean(axis=1, keepdims=True)
        cov = (Xc @ Xc.T) / max(1, Xc.shape[1]-1)
        # PSD clamp
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals_clamped = np.clip(eigvals, 1e-9, None)
            cov = (eigvecs * eigvals_clamped) @ eigvecs.T
        except Exception:
            pass
        std = np.sqrt(np.diag(cov))
        corr = np.zeros_like(cov)
        for i in range(len(std)):
            for j in range(len(std)):
                denom = std[i] * std[j]
                corr[i, j] = cov[i, j] / denom if denom > 0 else 0.0
        # cache
        self._corr_cache.clear()
        for i, a in enumerate(valid_symbols):
            for j, b in enumerate(valid_symbols):
                self._corr_cache[(a, b)] = float(corr[i, j])
        self._last_matrix_ts = self._last_update_ts

    def get_corr(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        # Recompute lazily if stale (> lookback updates since last build)
        if (self._last_update_ts - self._last_matrix_ts) >= 1:
            self._recompute_corr_matrix()
        return self._corr_cache.get((a, b), self._corr_cache.get((b, a), 0.0))

    # ------------- clusters -------------
    def get_cluster(self, symbol: str) -> Optional[str]:
        return self.cluster_map.get(symbol)

    def active_clusters(self, open_positions: Iterable[dict]) -> set[str]:
        clusters = set()
        for p in open_positions:
            try:
                c = self.get_cluster(p.get('symbol'))
                if c:
                    clusters.add(c)
            except Exception:
                continue
        return clusters

    # ------------- readiness -------------
    def ready(self, symbol: Optional[str] = None) -> bool:
        if symbol:
            dq = self._closes.get(symbol)
            return bool(dq and len(dq) >= self.min_bars)
        # global readiness if any symbol meets min_bars
        return any(len(dq) >= self.min_bars for dq in self._closes.values())

__all__ = ["RiskEstimator", "VolCorrSnapshot"]
