"""Router stub for low-latency mode.

This module provides a minimal router that selects a pre-warmed route
given a symbol and latency mode. It exposes a small API so the runner can
call into it without depending on network logic.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class Route:
    venue: str
    channel: str
    prebuilt_payload: bytes


class Router:
    def __init__(self, routing_table: Dict[str, Route]):
        self._table = routing_table

    def select(self, symbol: str, lowlat: bool = False) -> Route:
        # Very small selection logic: return entry if present, else first
        if symbol in self._table:
            return self._table[symbol]
        # fallback
        return next(iter(self._table.values()))


def make_default_router() -> Router:
    # prebuild a compact payload for demonstration
    r = {
        "BTC-USD": Route(venue="Coinbase", channel="orders", prebuilt_payload=b"BUY BTC-USD"),
        "ETH-USD": Route(venue="Binance", channel="orders", prebuilt_payload=b"BUY ETH-USD"),
    }
    return Router(r)


__all__ = ["Router", "make_default_router", "Route"]
