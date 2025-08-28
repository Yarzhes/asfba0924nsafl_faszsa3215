from __future__ import annotations
from typing import Dict, Any
from .http import HTTPClient
import time


class OKXAdapter:
    def __init__(self, rate_limit_rps: float = 5.0):
        self.client = HTTPClient(rate_limit_rps=rate_limit_rps)

    async def fetch_l2_orderbook(self, symbol: str, limit: int = 200) -> Dict[str, Any]:
        url = 'https://www.okx.com/api/v5/market/books'
        r = await self.client.get(url, params={'instId': symbol, 'sz': limit})
        j = r.json()
        ts = int(time.time()*1000)
        bids = []
        asks = []
        for item in j.get('data', [{}])[0].get('bids', []):
            bids.append((float(item[0]), float(item[1])))
        for item in j.get('data', [{}])[0].get('asks', []):
            asks.append((float(item[0]), float(item[1])))
        return {'bids': bids, 'asks': asks, 'ts': ts}

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        url = 'https://www.okx.com/api/v5/market/ticker'
        r = await self.client.get(url, params={'instId': symbol})
        j = r.json()
        ts = int(time.time()*1000)
        data = j.get('data', [{}])[0]
        return {'bid': float(data.get('bidPx', 0)), 'ask': float(data.get('askPx', 0)), 'ts': ts}

    async def fetch_funding(self, symbol: str) -> Dict[str, Any] | None:
        return None

    async def close(self):
        await self.client.close()
