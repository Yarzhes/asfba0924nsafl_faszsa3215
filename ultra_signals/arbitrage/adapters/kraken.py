from __future__ import annotations
from typing import Dict, Any
from .http import HTTPClient
import time


class KrakenAdapter:
    def __init__(self, rate_limit_rps: float = 2.0):
        self.client = HTTPClient(rate_limit_rps=rate_limit_rps)

    async def fetch_l2_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        # Kraken expects pair mapping; symbol should be provided in Kraken format
        url = 'https://api.kraken.com/0/public/Depth'
        r = await self.client.get(url, params={'pair': symbol, 'count': limit})
        j = r.json()
        ts = int(time.time()*1000)
        result = j.get('result', {})
        if not result:
            return {'bids': [], 'asks': [], 'ts': ts}
        # take first key
        first = next(iter(result.values()))
        bids = [(float(px), float(sz)) for px, sz, _ in first.get('bids', [])]
        asks = [(float(px), float(sz)) for px, sz, _ in first.get('asks', [])]
        return {'bids': bids, 'asks': asks, 'ts': ts}

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        url = 'https://api.kraken.com/0/public/Ticker'
        r = await self.client.get(url, params={'pair': symbol})
        j = r.json()
        ts = int(time.time()*1000)
        result = j.get('result', {})
        if not result:
            return {'bid': 0.0, 'ask': 0.0, 'ts': ts}
        first = next(iter(result.values()))
        bid = float(first.get('b', [0])[0])
        ask = float(first.get('a', [0])[0])
        return {'bid': bid, 'ask': ask, 'ts': ts}

    async def fetch_funding(self, symbol: str) -> Dict[str, Any] | None:
        return None

    async def close(self):
        await self.client.close()
