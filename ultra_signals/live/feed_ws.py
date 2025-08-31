"""WebSocket feed adapter (simplified, corrected indentation).

Forwards events from ``BinanceWSClient`` into an ``asyncio.Queue`` and (if
present) ingests them into an attached FeatureStore so downstream engine
filters have data access. Previous revisions accidentally left code at class
scope causing syntax / name errors. This version scopes everything properly.
"""
from __future__ import annotations

import asyncio
from loguru import logger

from ultra_signals.data.binance_ws import BinanceWSClient

try:  # pragma: no cover - best‑effort guard
    from ultra_signals.guards.live_guards import pre_tick_guard
except Exception:  # pragma: no cover
    def pre_tick_guard(*_a, **_k):  # type: ignore
        return None


class FeedAdapter:
    def __init__(self, settings, queue: asyncio.Queue, venue_router=None, venue_id: str | None = None):
        self.settings = settings
        self.queue = queue
        self._client = BinanceWSClient(settings)
        self._running = False
        self.venue_router = venue_router
        self._data_venue_id = venue_id or "binance_usdm"
        self._venue_books: dict[str, dict] = {}
        # Optional FeatureStore (attached later by LiveRunner)
        self.feature_store = None  # type: ignore

    async def run(self):  # pragma: no cover - network path
        self._running = True
        # Subscribe only to configured symbols/timeframes (client adds aux streams internally)
        timeframes = list(getattr(self.settings.runtime, "timeframes", []))
        symbols = list(getattr(self.settings.runtime, "symbols", []))
        try:
            self._client.subscribe(symbols, timeframes)
        except Exception as e:  # early failure
            logger.error(f"feed.subscribe_error symbols={symbols} tfs={timeframes} err={e}")
            return

        async for event in self._client.start():
            # 1. Stamp ingest time (best effort)
            if not hasattr(event, "_ingest_monotonic"):
                try:
                    setattr(event, "_ingest_monotonic", __import__("time").perf_counter())
                except Exception:
                    pass

            # 2. Pre‑tick guard
            try:
                pre_tick_guard(getattr(event, "symbol", "UNKNOWN"), self._data_venue_id.upper(), self.settings.model_dump() if hasattr(self.settings, "model_dump") else getattr(self.settings, "__dict__", {}))
            except Exception as e:
                logger.error(f"feed.dq_pre_tick_guard_block symbol={getattr(event,'symbol','?')} err={e}")
                continue

            # 3. Maintain venue book snapshot (for downstream spread/latency filters)
            try:
                if getattr(event, "event_type", "") == "bookTicker":
                    self._venue_books[self._data_venue_id.upper()] = {
                        "ts": getattr(event, "timestamp", 0),
                        "bid": getattr(event, "best_bid", None),
                        "ask": getattr(event, "best_ask", None),
                    }
                setattr(event, "_venue_books", dict(self._venue_books))
            except Exception:
                pass

            # 4. FeatureStore ingestion (best effort)
            if self.feature_store is not None:
                try:
                    self.feature_store.ingest_event(event)  # type: ignore[attr-defined]
                except Exception:
                    # swallow to keep feed resilient
                    pass

            # 5. Enqueue event (drop oldest closed-bar if queue full to preserve latest)
            try:
                self.queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    if getattr(event, "closed", False):
                        try:
                            self.queue.get_nowait()
                        except Exception:
                            pass
                        self.queue.put_nowait(event)
                except Exception:
                    pass

            # 6. Health bookkeeping
            if self.venue_router and self._data_venue_id:
                try:
                    self.venue_router.health.record_ws_staleness(self._data_venue_id, 0.0)
                except Exception:
                    pass

        logger.info("FeedAdapter stream ended. Subscriptions: {}", self._client._subscriptions)

    async def stop(self):  # pragma: no cover
        self._running = False
        try:
            await self._client.stop()
        except Exception:
            pass


__all__ = ["FeedAdapter"]
