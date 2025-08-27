"""Real-time notional and cluster exposure aggregation.

This module collects per-symbol long/short notionals and aggregates them to
clusters then derives portfolio beta via provided betas.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PortfolioExposure:
    cluster_map: Dict[str, str] = field(default_factory=dict)

    symbol_notionals: Dict[str, float] = field(default_factory=dict)  # signed
    cluster_notionals: Dict[str, float] = field(default_factory=dict)

    def update_position(self, symbol: str, notional: float) -> None:
        self.symbol_notionals[symbol] = float(notional)
        self._recompute_clusters()

    def remove_symbol(self, symbol: str) -> None:
        self.symbol_notionals.pop(symbol, None)
        self._recompute_clusters()

    def _recompute_clusters(self) -> None:
        clusters = {}
        for sym, notional in self.symbol_notionals.items():
            cl = self.cluster_map.get(sym, "_other")
            clusters[cl] = clusters.get(cl, 0.0) + float(notional)
        self.cluster_notionals = clusters

    def net_long_short(self) -> Dict[str, float]:
        long_net = sum(v for v in self.symbol_notionals.values() if v > 0)
        short_net = sum(-v for v in self.symbol_notionals.values() if v < 0)
        return {"long": float(long_net), "short": float(short_net)}

    def project_after_trade(self, symbol: str, add_notional: float) -> Dict[str, float]:
        curr = dict(self.symbol_notionals)
        curr[symbol] = curr.get(symbol, 0.0) + float(add_notional)
        return curr
