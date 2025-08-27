"""Bybit linear perpetual adapter (stub)."""
from __future__ import annotations
import time
import random
from typing import Any, Dict, List
from .base import ExchangeVenue, BookTop, OrderAck, CancelAck, Position, AccountInfo, TokenBucket


class BybitPerpPaper(ExchangeVenue):
    id = "bybit_perp"

    def __init__(self, symbol_mapper, rate_limit_rps: float = 8.0, dry_run: bool = True):
        self.mapper = symbol_mapper
        self.bucket = TokenBucket(rate_limit_rps)
        self.dry_run = dry_run
        self._orders: Dict[str, OrderAck] = {}
        self._positions: Dict[str, Position] = {}

    async def connect_streams(self, symbols: List[str], tfs: List[str]) -> None:  # pragma: no cover
        return

    async def fetch_ohlcv(self, symbol: str, tf: str, start, end) -> Any:  # pragma: no cover
        return []

    async def get_orderbook_top(self, symbol: str) -> BookTop:
        px = 50000.0 if symbol.startswith("BTC") else 2500.0
        # wider spread to allow deterministic preference in tests
        bid = px - 1.5
        ask = px + 1.5
        return BookTop(bid=bid, bid_size=4, ask=ask, ask_size=4, ts=int(time.time()*1000))

    async def place_order(self, plan: Dict[str, Any], client_order_id: str) -> OrderAck:
        await self.bucket.take()
        sym = plan.get("symbol")
        qty = abs(plan.get("qty", 1))
        side = plan.get("side")
        reduce_only = plan.get("reduce_only", False)
        force_partial = plan.get("force_partial", False)
        pos = self._positions.get(sym)
        cur_qty = pos.qty if pos else 0.0
        if side in ("LONG", "BUY"):
            candidate_qty = cur_qty + qty
            if reduce_only and cur_qty >= 0:
                candidate_qty = cur_qty
        elif side in ("SHORT", "SELL"):
            candidate_qty = cur_qty - qty
            if reduce_only and cur_qty <= 0:
                candidate_qty = cur_qty
        else:
            candidate_qty = 0
        new_qty = candidate_qty
        if new_qty == 0:
            self._positions.pop(sym, None)
        else:
            self._positions[sym] = Position(symbol=sym, qty=new_qty, avg_px=plan.get("price", 0))
        status = "FILLED"
        filled = qty
        if reduce_only and new_qty == cur_qty and qty > 0 and side in ("LONG","BUY","SHORT","SELL"):
            status = "REJECTED"
            filled = 0
        if force_partial and filled > 0:
            filled = max(0.0001, filled * 0.5)
            status = "PARTIAL"
        ack = OrderAck(client_order_id=client_order_id, venue_order_id=f"Y-{client_order_id[:10]}", status=status, filled_qty=filled if filled else None, avg_px=plan.get("price"))
        self._orders[client_order_id] = ack
        return ack

    async def amend_order(self, order_id: str, **kwargs) -> OrderAck:  # pragma: no cover
        return self._orders.get(order_id)  # type: ignore

    async def cancel_order(self, order_id: str, client_order_id: str) -> CancelAck:  # pragma: no cover
        return CancelAck(client_order_id=client_order_id, venue_order_id=order_id, status="CANCELED")

    async def positions(self) -> List[Position]:
        return list(self._positions.values())

    async def open_orders(self) -> List[OrderAck]:  # pragma: no cover
        return [o for o in self._orders.values() if o.status not in ("CANCELED",)]

    async def account(self) -> AccountInfo:  # pragma: no cover
        eq = 100000.0 + random.uniform(-120, 120)
        return AccountInfo(equity=eq, balance=eq)

    async def ping(self) -> float:
        return random.uniform(25, 70)

    def normalize_symbol(self, internal_symbol: str) -> str:
        return self.mapper.to_venue(internal_symbol, self.id)

__all__ = ["BybitPerpPaper"]
