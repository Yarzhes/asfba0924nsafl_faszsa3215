"""Async Bybit adapter (mock)."""
import asyncio
import time
from typing import Optional


class BybitAsyncAdapter:
    def __init__(self, engine, symbols=None):
        self.engine = engine
        self.symbols = symbols or []
        self._task = None

    async def _mock_feed(self, symbol: str, interval: float = 0.02, count: int = 1000):
        for i in range(count):
            t = int(time.time())
            price = 200.0 + (i % 7) * 0.2
            qty = 2 + (i % 4)
            side = 'buy' if i % 3 == 0 else 'sell'
            try:
                self.engine.ingest_trade(t, price, qty, side, aggressor=True)
            except Exception:
                pass
            await asyncio.sleep(interval)

    async def start(self):
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


__all__ = ['BybitAsyncAdapter']
