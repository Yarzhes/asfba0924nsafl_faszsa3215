"""WebSocket feed adapter (simplified).

Provides an async task that forwards events from ``BinanceWSClient`` into an
``asyncio.Queue`` after running lightweight data-quality guards. The previous
implementation accidentally referenced ``self`` at class scope which broke
test collection. This version confines all instance attribute assignments to
``__init__``.
"""
from __future__ import annotations
import asyncio
from loguru import logger
from ultra_signals.data.binance_ws import BinanceWSClient

try:  # pragma: no cover - guard import fallback
    from ultra_signals.guards.live_guards import pre_tick_guard
except Exception:  # pragma: no cover
    pre_tick_guard = lambda *a, **k: None  # type: ignore


class FeedAdapter:
    def __init__(self, settings, queue: asyncio.Queue, venue_router=None, venue_id: str | None = None):
        self.settings = settings
        self.queue = queue
        self._client = BinanceWSClient(settings)
        self._running = False
        self.venue_router = venue_router
        self._data_venue_id = venue_id or "binance_usdm"
        self._venue_books = {}  # per-venue latest bid/ask snapshot

    async def run(self):  # pragma: no cover (network path)
        self._running = True
        stream_types = getattr(self.settings.runtime, 'timeframes', []) + ["depth", "aggTrade"]
        self._client.subscribe(getattr(self.settings.runtime, 'symbols', []), stream_types)
        async for event in self._client.start():
            # Stamp ingest monotonic time for latency measurement if not present
            try:
                if not hasattr(event, "_ingest_monotonic"):
                    setattr(event, "_ingest_monotonic", __import__("time").perf_counter())
            except Exception:
                pass
            # DQ guard (tick-level)
            try:
                pre_tick_guard(getattr(event, 'symbol', 'UNKNOWN'), self._data_venue_id.upper(), self.settings.model_dump() if hasattr(self.settings,'model_dump') else getattr(self.settings,'__dict__', {}))
            except Exception as e:
                logger.error(f"feed.dq_pre_tick_guard_block symbol={getattr(event,'symbol','?')} err={e}")
                continue
            try:
                etype = getattr(event, 'event_type', '')
                if etype == 'bookTicker':
                    self._venue_books[self._data_venue_id.upper()] = {
                        'ts': getattr(event, 'timestamp', 0),
                        'bid': getattr(event, 'best_bid', None),
                        'ask': getattr(event, 'best_ask', None),
                    }
                setattr(event, '_venue_books', dict(self._venue_books))
                self.queue.put_nowait(event)
                if self.venue_router and self._data_venue_id:
                    try:
                        self.venue_router.health.record_ws_staleness(self._data_venue_id, 0.0)
                    except Exception:
                        pass
            except asyncio.QueueFull:
                try:
                    if getattr(event, 'closed', False):
                        try:
                            self.queue.get_nowait()
                        except Exception:
                            pass
                        self.queue.put_nowait(event)
                except Exception:
                    pass
        logger.info("FeedAdapter stream ended.")

    async def stop(self):  # pragma: no cover
        self._running = False
        await self._client.stop()

__all__ = ["FeedAdapter"]
