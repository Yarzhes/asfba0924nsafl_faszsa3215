"""Async OKX adapter (mock)."""
import asyncio
import time
from typing import Optional


class OKXAsyncAdapter:
    def __init__(self, engine, symbols=None):
        self.engine = engine
        self.symbols = symbols or []
        self._task = None

    async def _mock_feed(self, symbol: str, interval: float = 0.015, count: int = 1000):
        for i in range(count):
            t = int(time.time())
            price = 300.0 + (i % 4) * 0.05
            qty = 1.5 + (i % 2)
            side = 'buy' if i % 4 < 2 else 'sell'
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


__all__ = ['OKXAsyncAdapter']
