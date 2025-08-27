"""Base protocol + shared dataclasses for venue adapters.

Adapters should stay < ~300 LOC. Shared concerns (signing, retries, throttling) can
graduate into helper modules later; for now only a token bucket skeleton is provided.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any, Optional, Dict, List
import time
import asyncio


@dataclass
class BookTop:
    bid: float
    bid_size: float
    ask: float
    ask_size: float
    ts: int


@dataclass
class OrderAck:
    client_order_id: str
    venue_order_id: str
    status: str  # NEW|FILLED|PARTIAL|REJECTED|CANCELED
    filled_qty: float | None = None
    avg_px: float | None = None
    raw: Any | None = None


@dataclass
class CancelAck:
    client_order_id: str
    venue_order_id: str
    status: str  # CANCELED|NOT_FOUND
    raw: Any | None = None


@dataclass
class Position:
    symbol: str
    qty: float
    avg_px: float
    unrealized_pnl: float | None = None
    raw: Any | None = None


@dataclass
class AccountInfo:
    equity: float
    balance: float | None = None
    margin_used: float | None = None
    raw: Any | None = None


class ExchangeVenue(Protocol):
    id: str

    async def connect_streams(self, symbols: List[str], tfs: List[str]) -> None: ...
    async def fetch_ohlcv(self, symbol: str, tf: str, start, end) -> Any: ...  # DataFrame-like
    async def get_orderbook_top(self, symbol: str) -> BookTop: ...
    async def place_order(self, plan: Dict[str, Any], client_order_id: str) -> OrderAck: ...
    async def amend_order(self, order_id: str, **kwargs) -> OrderAck: ...
    async def cancel_order(self, order_id: str, client_order_id: str) -> CancelAck: ...
    async def positions(self) -> List[Position]: ...
    async def account(self) -> AccountInfo: ...
    async def ping(self) -> float: ...  # ms
    def normalize_symbol(self, internal_symbol: str) -> str: ...
    async def open_orders(self) -> List[OrderAck]: ...


class TokenBucket:
    """Simple token bucket for REST rate limiting (coarse, cooperative)."""
    def __init__(self, rate_per_sec: float, capacity: Optional[float] = None):
        self.rate = float(rate_per_sec)
        self.capacity = capacity or self.rate
        self.tokens = self.capacity
        self.last = time.time()
        self._lock = asyncio.Lock()

    async def take(self, amount: float = 1.0):
        async with self._lock:
            while True:
                now = time.time()
                elapsed = now - self.last
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                    self.last = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                await asyncio.sleep(0.01)


__all__ = [
    "ExchangeVenue",
    "BookTop",
    "OrderAck",
    "CancelAck",
    "Position",
    "AccountInfo",
    "TokenBucket",
]
