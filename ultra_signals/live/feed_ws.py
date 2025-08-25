"""Feed adaptor pushing normalized events into a bounded queue.

Currently proxies the existing `BinanceWSClient` interface. For unit tests we
can inject synthetic events directly.
"""
from __future__ import annotations
import asyncio
from loguru import logger
from ultra_signals.data.binance_ws import BinanceWSClient


class FeedAdapter:
    def __init__(self, settings, queue: asyncio.Queue):
        self.settings = settings
        self.queue = queue
        self._client = BinanceWSClient(settings)
        self._running = False

    async def run(self):  # pragma: no cover (network path)
        self._running = True
        stream_types = self.settings.runtime.timeframes + ["depth", "aggTrade"]
        self._client.subscribe(self.settings.runtime.symbols, stream_types)
        async for event in self._client.start():
            # Stamp ingest monotonic time for latency measurement if not present
            try:
                if not hasattr(event, "_ingest_monotonic"):
                    setattr(event, "_ingest_monotonic", __import__("time").perf_counter())
            except Exception:
                pass
            try:
                self.queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop non-critical events: anything that's not a closed kline
                try:
                    closed = getattr(event, "closed", False)
                    if closed:
                        # force space by removing one oldest non-critical (simple strategy)
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
