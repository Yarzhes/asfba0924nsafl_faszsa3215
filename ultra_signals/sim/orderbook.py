"""Synthetic orderbook model for BrokerSim (Sprint 36 minimal version).

Builds a micro orderbook ladder from bar data heuristics:
- Spread derived from ATR percentile proxy (if provided) else fixed bps
- Depth distribution geometric across levels
- Advance() refreshes a fraction of consumed depth each ms

This is intentionally lightweight; refine with real L2 ingestion later.
"""
from __future__ import annotations
from typing import List, Dict, Any
import random, math

class SyntheticOrderBook:
    def __init__(self, symbol: str, levels: int=10, seed: int=42, base_spread_bps: float=5.0):
        self.symbol = symbol
        self.levels = levels
        self.rng = random.Random(seed)
        self.base_spread_bps = base_spread_bps
        self.mid_price = 0.0
        self._ladder: List[tuple] = []  # ask side for BUY sweeps (price asc, qty)
        self._bid_ladder: List[tuple] = []  # for SELL sweeps

    def rebuild_from_bar(self, bar: Dict[str, Any]):
        close = float(bar.get('close'))
        high = float(bar.get('high', close)); low = float(bar.get('low', close))
        self.mid_price = close
        tr = max(1e-9, high - low)
        spread_bp = self.base_spread_bps * (0.5 + self.rng.random())
        spread_px = self.mid_price * spread_bp / 10_000.0
        best_ask = self.mid_price + spread_px/2
        best_bid = self.mid_price - spread_px/2
        # geometric depth curve
        total_depth = max(1.0, tr / max(1e-9, spread_px))  # crude liquidity proxy
        top_qty = max(0.1, total_depth/ (self.levels*1.5))
        asks=[]; bids=[]
        for i in range(self.levels):
            step = (i+1)
            px_a = best_ask + step * spread_px * 0.2
            px_b = best_bid - step * spread_px * 0.2
            decay = math.exp(-0.35*step)
            qty = top_qty * (0.6 + 0.8*self.rng.random()) * decay
            asks.append((px_a, qty))
            bids.append((px_b, qty))
        self._ladder = asks
        self._bid_ladder = bids

    def best_bid(self) -> float:
        return self._bid_ladder[0][0] if self._bid_ladder else (self.mid_price*0.999)
    def best_ask(self) -> float:
        return self._ladder[0][0] if self._ladder else (self.mid_price*1.001)

    def ladder(self) -> List[tuple]:
        return list(self._ladder)
    def ladder_bid(self) -> List[tuple]:
        return list(self._bid_ladder)

    def advance(self, ms: int):
        # simple replenishment: random small noise to each level
        if not self._ladder: return
        for idx,(px,qty) in enumerate(self._ladder):
            # 1% chance large replenish, else small drift
            if self.rng.random() < 0.01:
                qty = qty * (0.8 + 0.8*self.rng.random())
            else:
                qty = qty * (0.98 + 0.04*self.rng.random())
            self._ladder[idx] = (px, max(qty, 1e-6))
        for idx,(px,qty) in enumerate(self._bid_ladder):
            if self.rng.random() < 0.01:
                qty = qty * (0.8 + 0.8*self.rng.random())
            else:
                qty = qty * (0.98 + 0.04*self.rng.random())
            self._bid_ladder[idx] = (px, max(qty, 1e-6))

__all__ = ['SyntheticOrderBook']
