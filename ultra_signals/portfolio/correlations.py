"""Rolling correlation & beta estimation with simple shrinkage.

Designed to be lightweight and dependency-minimal (pure pandas/numPy) so it can
run every bar without large overhead. We keep only the last `lookback` rows of
price/return history per symbol.

Interfaces kept small so integration into existing engine is low risk.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Deque, List, Optional, Tuple
from collections import deque
import math
import pandas as pd
import numpy as np


@dataclass
class RollingCorrelationBeta:
    leader: str
    lookback: int
    shrinkage_lambda: float = 0.0  # 0 == no shrinkage, else ridge blend
    max_symbols: int = 128

    # internal state
    _prices: Dict[str, Deque[float]] = field(default_factory=dict)
    _returns_cache: Optional[pd.DataFrame] = None
    _last_ts: Optional[int] = None
    betas: Dict[str, float] = field(default_factory=dict)
    correlations: Dict[Tuple[str, str], float] = field(default_factory=dict)
    median_abs_corr_to_leader: float = 0.0
    high_corr_regime: bool = False
    corr_threshold_high: float = 0.6

    def update_price(self, symbol: str, price: float, ts: int) -> None:
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=self.lookback + 1)
        self._prices[symbol].append(float(price))
        self._last_ts = ts

    # -----------------------
    # Returns & matrix update
    # -----------------------
    def _compute_returns_df(self) -> pd.DataFrame:
        data = {}
        for sym, dq in self._prices.items():
            if len(dq) < 2:
                continue
            # log returns
            arr = np.array(dq, dtype=float)
            r = np.diff(np.log(arr))
            if len(r) >= self.lookback:
                r = r[-self.lookback :]
            data[sym] = r
        if not data:
            return pd.DataFrame()
        # align lengths by padding front with NaN then trimming
        max_len = max(len(v) for v in data.values())
        for k, v in data.items():
            if len(v) < max_len:
                data[k] = np.concatenate([np.full(max_len - len(v), np.nan), v])
        df = pd.DataFrame(data)
        df = df.dropna(how="any")  # require full row
        return df.tail(self.lookback)

    def recompute(self) -> None:
        df = self._compute_returns_df()
        if df.empty or self.leader not in df.columns:
            return
        cov = df.cov()
        if self.shrinkage_lambda > 0:
            lam = float(self.shrinkage_lambda)
            diag = np.diag(np.diag(cov.values))
            cov_values = (1 - lam) * cov.values + lam * diag
            cov = pd.DataFrame(cov_values, index=cov.index, columns=cov.columns)
        std = np.sqrt(np.diag(cov.values))
        corr = cov.copy()
        for i, a in enumerate(cov.index):
            for j, b in enumerate(cov.columns):
                denom = std[i] * std[j]
                corr.iloc[i, j] = cov.iloc[i, j] / denom if denom > 0 else 0.0
                self.correlations[(a, b)] = float(corr.iloc[i, j])
        # betas vs leader
        var_leader = float(cov.loc[self.leader, self.leader])
        if var_leader <= 0:
            return
        new_betas = {}
        for sym in cov.columns:
            if sym == self.leader:
                new_betas[sym] = 1.0
            else:
                new_betas[sym] = float(cov.loc[sym, self.leader] / var_leader)
        self.betas = new_betas
        # regime flag
        abs_corrs = [abs(self.correlations.get((sym, self.leader), 0.0)) for sym in df.columns if sym != self.leader]
        if abs_corrs:
            self.median_abs_corr_to_leader = float(np.median(abs_corrs))
            self.high_corr_regime = self.median_abs_corr_to_leader >= self.corr_threshold_high
        else:
            self.median_abs_corr_to_leader = 0.0
            self.high_corr_regime = False

    def get_beta(self, symbol: str) -> float:
        return float(self.betas.get(symbol, 0.0))

    def portfolio_beta(self, notionals: Dict[str, float], equity: float) -> float:
        if equity <= 0:
            return 0.0
        total = 0.0
        for sym, notional in notionals.items():
            beta = self.get_beta(sym)
            total += (notional / equity) * beta
        return float(total)
