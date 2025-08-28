"""Event-driven tick replayer with simple queue model.

This module is a minimal, deterministic starting point for Sprint 63. It replays
trade and depth events in timestamp order, feeds a BookRebuilder, and simulates
FIFO fills for incoming aggressive trades against the top-of-book.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Iterable
import heapq
import math
import random

from ultra_signals.market.book_rebuilder import BookRebuilder


class QueuePositionEstimator:
    """Estimate fills based on resting size and queue position.

    Modes:
     - fifo: first-in-first-out (default)
     - pro_rata: distribute aggressor across available levels proportionally
    """
    def __init__(self, mode: str = "fifo"):
        self.mode = mode

    def match(self, book_side: Dict[float, float], size: float) -> List[Dict[str, float]]:
        remaining = float(size)
        fills: List[Dict[str, float]] = []
        if self.mode == "fifo":
            for px in sorted(book_side.keys(), reverse=(False if book_side and min(book_side.keys()) > 0 else False)):
                if remaining <= 0:
                    break
                avail = book_side.get(px, 0.0)
                take = min(avail, remaining)
                if take > 0:
                    fills.append({"px": px, "qty": take})
                    remaining -= take
        elif self.mode == "pro_rata":
            # proportional across price levels based on available size
            total = sum(book_side.values())
            if total <= 0:
                return []
            for px, avail in book_side.items():
                take = min(remaining, size * (avail / total))
                if take > 0:
                    fills.append({"px": px, "qty": take})
                    remaining -= take
                    if remaining <= 0:
                        break
        else:
            # fallback to FIFO
            return self.match(book_side, size)
        return fills


class LatencyModel:
    def __init__(self, wire_ms: int = 1, match_ms: int = 1, loss_prob: float = 0.0):
        self.wire_ms = int(wire_ms)
        self.match_ms = int(match_ms)
        self.loss_prob = float(loss_prob)
        self.rng = random.Random(42)

    def should_drop(self) -> bool:
        return self.rng.random() < self.loss_prob

    def total_delay_ms(self) -> int:
        return self.wire_ms + self.match_ms


class TickReplayer:
    def __init__(self, max_levels: int = 50, seed: int = 42, estimator_mode: str = "fifo", latency: Optional[LatencyModel] = None):
        self.book = BookRebuilder(max_levels=max_levels)
        self._events_heap: List[tuple] = []  # (ts, idx, event)
        self._counter = 0
        self.estimator = QueuePositionEstimator(mode=estimator_mode)
        self.latency = latency or LatencyModel()
        self._rng = random.Random(seed)

    def add_event(self, event: Dict[str, Any]) -> None:
        """Add event dict with required field 'ts' (int milliseconds) and 'type' ('snapshot'|'delta'|'trade')."""
        ts = int(event.get("ts") or 0)
        heapq.heappush(self._events_heap, (ts, self._counter, event))
        self._counter += 1

    def feed_from_iter(self, it: Iterable[Dict[str, Any]]):
        for row in it:
            self.add_event(row)

    def replay(self) -> List[Dict[str, Any]]:
        """Replay all events in order. Returns list of fills simulated for trade events."""
        fills: List[Dict[str, Any]] = []
        while self._events_heap:
            ts, _, ev = heapq.heappop(self._events_heap)
            typ = ev.get("type")
            if typ == "snapshot":
                self.book.load_snapshot(ev.get("data") or {})
            elif typ == "delta":
                d = ev.get("data") or {}
                # attach provenance
                d["src"] = ev.get("src") or ev.get("source") or d.get("src")
                self.book.apply_delta(d)
            elif typ == "trade":
                # schedule an execution fill event at ts + latency (to simulate wire + match delays)
                if self.latency.should_drop():
                    fills.append({"ts": ts, "dropped": True, "reason": "latency_loss"})
                    continue
                delay_ms = self.latency.total_delay_ms()
                exec_ts = int(ts) + int(delay_ms)
                # push delayed exec event
                heapq.heappush(self._events_heap, (exec_ts, self._counter, {"type": "exec_fill", "orig": ev}))
                self._counter += 1
            elif typ == "exec_fill":
                orig = ev.get("orig") or {}
                side = orig.get("side")
                size = float(orig.get("size") or 0)
                price = float(orig.get("price") or 0)
                fills.append(self._simulate_fill(ts, side, size, price))
            elif typ == "cancel":
                # cancel data may contain price and size or cancel_all
                data = ev.get("data") or {}
                side = data.get("side")
                if data.get("cancel_all"):
                    if side == "buy":
                        self.book._bids.clear()
                    elif side == "sell":
                        self.book._asks.clear()
                else:
                    price = data.get("price")
                    size = float(data.get("size") or 0)
                    if price is not None:
                        pxf = float(price)
                        if side == "buy":
                            # reduce bids at that price
                            if pxf in self.book._bids:
                                self.book._bids[pxf] = max(0.0, self.book._bids[pxf] - size)
                                if self.book._bids[pxf] <= 0:
                                    self.book._bids.pop(pxf, None)
                        else:
                            if pxf in self.book._asks:
                                self.book._asks[pxf] = max(0.0, self.book._asks[pxf] - size)
                                if self.book._asks[pxf] <= 0:
                                    self.book._asks.pop(pxf, None)
            else:
                continue
        return fills

    def _simulate_fill(self, ts: int, side: str, size: float, price: float) -> Dict[str, Any]:
        # Use estimator to compute fills, update book accordingly
        filled_qty = 0.0
        fill_events: List[Dict[str, float]] = []
        if side == "buy":
            # match against asks ascending
            book_side = {px: self.book._asks[px] for px in sorted(self.book._asks.keys())}
            parts = self.estimator.match(book_side, size)
            for p in parts:
                px = float(p["px"])
                qty = float(p["qty"])
                avail = self.book._asks.get(px, 0.0)
                taken = min(avail, qty)
                if taken <= 0:
                    continue
                self.book._asks[px] = max(0.0, avail - taken)
                if self.book._asks[px] <= 0:
                    self.book._asks.pop(px, None)
                filled_qty += taken
                fill_events.append({"px": px, "qty": taken})
        elif side == "sell":
            book_side = {px: self.book._bids[px] for px in sorted(self.book._bids.keys(), reverse=True)}
            parts = self.estimator.match(book_side, size)
            for p in parts:
                px = float(p["px"])
                qty = float(p["qty"])
                avail = self.book._bids.get(px, 0.0)
                taken = min(avail, qty)
                if taken <= 0:
                    continue
                self.book._bids[px] = max(0.0, avail - taken)
                if self.book._bids[px] <= 0:
                    self.book._bids.pop(px, None)
                filled_qty += taken
                fill_events.append({"px": px, "qty": taken})

        return {
            "ts": ts,
            "side": side,
            "requested_qty": size,
            "filled_qty": filled_qty,
            "fills": fill_events,
            "price": price,
            "micro": self.book.microprice(),
            "needs_resync": self.book.needs_resync(),
            "latency_ms": self.latency.total_delay_ms(),
        }


__all__ = ["TickReplayer", "QueuePositionEstimator", "LatencyModel"]
