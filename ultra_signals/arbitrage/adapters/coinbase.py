from __future__ import annotations
from typing import Dict, Any
from .http import HTTPClient
import time


class CoinbaseAdapter:
    def __init__(self, rate_limit_rps: float = 3.0):
        self.client = HTTPClient(rate_limit_rps=rate_limit_rps)

    async def fetch_l2_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        # Coinbase Pro style symbol like BTC-USD or BTC-USDT depending on mapping
        url = f'https://api.exchange.coinbase.com/products/{symbol}/book'
        r = await self.client.get(url, params={'level': 2})
        j = r.json()
        ts = int(time.time()*1000)
        bids = [(float(px), float(sz)) for px, sz, _ in j.get('bids', [])]
        asks = [(float(px), float(sz)) for px, sz, _ in j.get('asks', [])]
        return {'bids': bids, 'asks': asks, 'ts': ts}

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        url = f'https://api.exchange.coinbase.com/products/{symbol}/ticker'
        r = await self.client.get(url)
        j = r.json()
        ts = int(time.time()*1000)
        return {'bid': float(j.get('bid', 0)), 'ask': float(j.get('ask', 0)), 'ts': ts}

    async def fetch_funding(self, symbol: str) -> Dict[str, Any] | None:
        return None

    async def close(self):
        await self.client.close()
