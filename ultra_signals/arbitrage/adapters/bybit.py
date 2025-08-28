from __future__ import annotations
from typing import Dict, Any
from .http import HTTPClient
import time


class BybitAdapter:
    def __init__(self, rate_limit_rps: float = 5.0):
        self.client = HTTPClient(rate_limit_rps=rate_limit_rps)

    async def fetch_l2_orderbook(self, symbol: str, limit: int = 200) -> Dict[str, Any]:
        # Bybit public depth endpoint
        url = 'https://api.bybit.com/v2/public/orderBook/L2'
        r = await self.client.get(url, params={'symbol': symbol})
        j = r.json()
        ts = int(time.time()*1000)
        bids = []
        asks = []
        for item in j.get('result', []):
            px = float(item.get('price', 0))
            sz = float(item.get('size', 0))
            side = item.get('side')
            if side == 'Buy':
                bids.append((px, sz))
            else:
                asks.append((px, sz))
        bids.sort(key=lambda x: -x[0])
        asks.sort(key=lambda x: x[0])
        return {'bids': bids, 'asks': asks, 'ts': ts}

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        url = 'https://api.bybit.com/v2/public/tickers'
        r = await self.client.get(url, params={'symbol': symbol})
        j = r.json()
        ts = int(time.time()*1000)
        res = j.get('result')
        if not res:
            return {'bid': 0.0, 'ask': 0.0, 'ts': ts}
        r0 = res[0]
        return {'bid': float(r0.get('bid_price', 0)), 'ask': float(r0.get('ask_price', 0)), 'ts': ts}

    async def fetch_funding(self, symbol: str) -> Dict[str, Any] | None:
        return None

    async def close(self):
        await self.client.close()
