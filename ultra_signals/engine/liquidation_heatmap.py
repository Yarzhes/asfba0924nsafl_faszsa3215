from __future__ import annotations
"""
Sprint 15: Liquidation Heatmap (mock implementation)
---------------------------------------------------
Provides coarse liquidation cluster awareness for dynamic sizing.
In production this would query an API / precomputed map.
"""
from typing import List, Dict, Any
import bisect

class LiquidationHeatmap:
    def __init__(self, settings: dict):
        self.settings = settings or {}
        cfg = (self.settings.get('liquidation_heatmap') or {}) if isinstance(self.settings, dict) else {}
        self.min_cluster_usd = float(cfg.get('min_cluster_usd', 1_000_000))

    def get_liq_levels(self, symbol: str) -> List[Dict[str, Any]]:
        """Return list of liquidation clusters (price, side, size).
        Mock returns two synthetic clusters so sizing logic can exercise.
        side: 'long_liq' means longs liquidated there (bullish reversal potential)
              'short_liq' means shorts liquidated there (bearish reversal potential)
        """
        # In reality you'd look up symbol-specific data
        return [
            {"price": 20250.0, "side": "long_liq", "size": 3_200_000},
            {"price": 19800.0, "side": "short_liq", "size": 2_500_000},
        ]

    @staticmethod
    def compute_liq_risk(current_price: float, clusters: List[Dict[str, Any]], min_cluster_usd: float) -> float:
        """Compute a dimensionless risk metric based on distance to nearest large cluster.
        risk ~ size / (distance_pct * K). Clamped and scaled.
        Returns 0 (low) .. ~3 (very high) typical.
        """
        if current_price <= 0 or not clusters:
            return 0.0
        best = None
        for c in clusters:
            try:
                size = float(c.get('size', 0))
                if size < min_cluster_usd:
                    continue
                price = float(c.get('price'))
                dist_pct = abs(price - current_price) / current_price
                if dist_pct == 0:
                    dist_pct = 0.0001
                score = size / (dist_pct * min_cluster_usd)
                if best is None or score > best:
                    best = score
            except Exception:
                continue
        if best is None:
            return 0.0
        # Compress with log for stability
        import math
        return round(min(3.0, math.log(1 + best)), 4)
