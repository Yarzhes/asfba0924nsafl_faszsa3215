from __future__ import annotations
from typing import Dict, Any
from .http import HTTPClient
import time


class BinanceAdapter:
    def __init__(self, rate_limit_rps: float = 5.0):
        self.client = HTTPClient(rate_limit_rps=rate_limit_rps)

    async def fetch_l2_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        # Binance uses symbol like BTCUSDT and depth endpoint
        url = 'https://api.binance.com/api/v3/depth'
        r = await self.client.get(url, params={'symbol': symbol, 'limit': limit})
        j = r.json()
        ts = int(time.time()*1000)
        bids = [(float(px), float(sz)) for px, sz in j.get('bids', [])]
        asks = [(float(px), float(sz)) for px, sz in j.get('asks', [])]
        return {'bids': bids, 'asks': asks, 'ts': ts}

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        url = 'https://api.binance.com/api/v3/ticker/bookTicker'
        r = await self.client.get(url, params={'symbol': symbol})
        j = r.json()
        ts = int(time.time()*1000)
        return {'bid': float(j.get('bidPrice', 0)), 'ask': float(j.get('askPrice', 0)), 'ts': ts}

    async def fetch_funding(self, symbol: str) -> Dict[str, Any] | None:
        # Binance USDM and COIN-M have funding endpoints on different host; for public spot/perp detection
        # we'll skip funding here; funding provider covers it.
        return None

    async def close(self):
        await self.client.close()
