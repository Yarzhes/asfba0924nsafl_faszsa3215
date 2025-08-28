"""Async HTTP helpers and a tiny token-bucket for rate limiting."""
from typing import Optional
import asyncio, time
import httpx


class AsyncTokenBucket:
    def __init__(self, rate: float, capacity: float = None):
        self.rate = rate
        self.capacity = capacity or rate
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def take(self, tokens: float = 1.0):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            if tokens <= self._tokens:
                self._tokens -= tokens
                return
            # wait until tokens available
            needed = tokens - self._tokens
            wait = needed / self.rate
        await asyncio.sleep(wait)


class HTTPClient:
    def __init__(self, rate_limit_rps: float = 5.0, timeout: int = 10):
        self._client = httpx.AsyncClient(timeout=timeout)
        self._bucket = AsyncTokenBucket(rate_limit_rps)

    async def get(self, url: str, params: Optional[dict] = None) -> httpx.Response:
        await self._bucket.take(1.0)
        return await self._client.get(url, params=params)

    async def close(self):
        await self._client.aclose()
