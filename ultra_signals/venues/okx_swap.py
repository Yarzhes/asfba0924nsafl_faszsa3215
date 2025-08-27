"""OKX swap adapter (stub)."""
from __future__ import annotations
import time, random
from typing import Dict, Any, List
from .base import ExchangeVenue, BookTop, OrderAck, CancelAck, Position, AccountInfo, TokenBucket


class OKXSwapPaper(ExchangeVenue):
    id = "okx_swap"

    def __init__(self, symbol_mapper, rate_limit_rps: float = 6.0, dry_run: bool = True):
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
        return BookTop(bid=px-1.2, bid_size=3.5, ask=px+1.2, ask_size=3.3, ts=int(time.time()*1000))

    async def place_order(self, plan: Dict[str, Any], client_order_id: str) -> OrderAck:
        await self.bucket.take()
        sym = plan.get("symbol")
        qty = abs(plan.get("qty", 1))
        side = plan.get("side")
        pos = self._positions.get(sym)
        if side in ("LONG","BUY"):
            new_qty = (pos.qty if pos else 0) + qty
        elif side in ("SHORT","SELL"):
            new_qty = (pos.qty if pos else 0) - qty
        else:
            new_qty = 0
        if new_qty == 0:
            self._positions.pop(sym, None)
        else:
            self._positions[sym] = Position(symbol=sym, qty=new_qty, avg_px=plan.get("price",0))
        ack = OrderAck(client_order_id=client_order_id, venue_order_id=f"O-{client_order_id[:10]}", status="FILLED", filled_qty=qty, avg_px=plan.get("price"))
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
        eq = 100000 + random.uniform(-80, 80)
        return AccountInfo(equity=eq, balance=eq)

    async def ping(self) -> float:
        return random.uniform(30, 75)

    def normalize_symbol(self, internal_symbol: str) -> str:
        return self.mapper.to_venue(internal_symbol, self.id)

__all__ = ["OKXSwapPaper"]