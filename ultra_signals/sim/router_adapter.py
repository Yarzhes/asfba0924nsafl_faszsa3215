"""Router adapter to swap fast_router with BrokerSim (Sprint 36).

Provides execute_fast_order(...) signature returning a FastExecResult-like object
so existing integration points can switch based on config.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal
from .broker import BrokerSim, Order, map_side
from .orderbook import SyntheticOrderBook

@dataclass
class SimExecResult:
    accepted: bool
    reason: str
    venue: Optional[str] = None
    order: Optional[Dict[str, Any]] = None
    expected_price: Optional[float] = None
    spread_bps: Optional[float] = None
    depth_ok: Optional[bool] = None
    retries: int = 0

class BrokerRouterAdapter:
    def __init__(self, settings: Dict[str, Any]):
        sim_cfg = (settings.get('broker_sim') or {})
        seed = int(sim_cfg.get('rng_seed', 42))
        # Single venue synthetic for now (extend to multi by symbol mapping later)
        ob = SyntheticOrderBook('GENERIC', levels=((sim_cfg.get('orderbook') or {}).get('levels',10)))
        self.sim = BrokerSim(sim_cfg, ob, rng_seed=seed)
        self.settings = settings

    def execute_fast_order(self, *, symbol: str, side: Literal['LONG','SHORT'], size: float, price: float|None, settings: Dict[str, Any], quotes: Optional[Dict[str, Dict[str,float]]]=None) -> SimExecResult:
        # rebuild orderbook from current price context (best-effort)
        bar = {'close': price, 'high': price*1.001 if price else 0, 'low': price*0.999 if price else 0}
        self.sim.orderbook.rebuild_from_bar(bar)
        internal_side = map_side(side)
        order = Order(id=f"SIM-{symbol}-{self.sim.clock.now()}", symbol=symbol, side=internal_side, type='MARKET', qty=float(size))
        fills = self.sim.submit_order(order)
        if not fills:
            return SimExecResult(False, 'NO_FILL', venue=self.sim.venue)
        # aggregate fill metrics
        total_qty = sum(f.qty for f in fills)
        vwap = sum(f.price*f.qty for f in fills)/total_qty if total_qty else None
        return SimExecResult(True, 'OK', venue=self.sim.venue, order={'symbol': symbol, 'side': side, 'qty': size, 'price': vwap, 'order_type': 'MARKET'}, expected_price=vwap, spread_bps=None, depth_ok=True)

__all__ = ['BrokerRouterAdapter','SimExecResult']
