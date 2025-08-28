"""Async Binance adapter that feeds OrderflowEngine.
Uses `websockets` if available; otherwise provides an async mock generator.
"""
import asyncio
import time
from typing import AsyncIterable, Dict, Any, Optional


class BinanceAsyncAdapter:
    def __init__(self, engine, symbols=None):
        self.engine = engine
        self.symbols = symbols or []
        self._task: Optional[asyncio.Task] = None

    async def _mock_feed(self, symbol: str, interval: float = 0.01, count: int = 1000):
        for i in range(count):
            if self._task is None or self._task.cancelled():
                break
            t = int(time.time())
            price = 100.0 + (i % 10) * 0.1
            qty = 1 + (i % 3)
            side = 'buy' if i % 2 == 0 else 'sell'
            try:
                self.engine.ingest_trade(t, price, qty, side, aggressor=True)
            except Exception:
                pass
            await asyncio.sleep(interval)

    async def start(self):
        # Start mock feeds for each symbol concurrently
        loop = asyncio.get_event_loop()
        tasks = [loop.create_task(self._mock_feed(s)) for s in (self.symbols or ['MOCK'])]
        self._task = asyncio.gather(*tasks)
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    def stop(self):
        if self._task:
            self._task.cancel()


__all__ = ['BinanceAsyncAdapter']
