from __future__ import annotations
from typing import Dict

class DepthAggregator:
    """
    Thinness heuristic. For tests + backtests we keep defaults lenient:
    - Only use top_qty threshold if it's explicitly set > 0 in settings.
    """
    def __init__(self, cfg: dict):
        dcfg = cfg.get("alpha", {}).get("depth", {})
        self.thin_spread_bps = float(dcfg.get("thin_spread_bps", 12))  # a bit looser than 10
        self.min_top_qty = float(dcfg.get("min_top_qty", 0.0))         # IMPORTANT: default 0 disables qty check

    def evaluate(self, best_bid: float, best_ask: float, bid_qty: float, ask_qty: float) -> Dict[str, float]:
        mid = (best_bid + best_ask) / 2.0 if best_bid and best_ask else 0.0
        spread_bps = ( (best_ask - best_bid) / mid * 10000.0 ) if mid else 0.0
        top_tot = float((bid_qty or 0.0) + (ask_qty or 0.0))

        # Only activate qty threshold if user configured a positive threshold
        qty_thin = (self.min_top_qty > 0.0 and top_tot < self.min_top_qty)
        spread_thin = spread_bps > self.thin_spread_bps

        thin = 1.0 if (spread_thin or qty_thin) else 0.0
        return {"spread_bps": spread_bps, "top_qty": top_tot, "is_thin": thin}
