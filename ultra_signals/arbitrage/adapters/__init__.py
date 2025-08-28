"""Public exchange adapters (REST/WS scaffolds).

Adapters provide async methods:
  - fetch_l2_orderbook(symbol, limit) -> {'bids':[(px,size)...], 'asks':[...] , 'ts': ms}
  - fetch_ticker(symbol) -> {'bid':, 'ask':, 'ts':ms}
  - fetch_funding(symbol) -> {'funding_rate': float, 'funding_time': ms}

These are minimal, use public endpoints only, and include simple rate limiting wrappers.
"""

from .binance import BinanceAdapter
from .bybit import BybitAdapter
from .okx import OKXAdapter
from .coinbase import CoinbaseAdapter
from .kraken import KrakenAdapter

__all__ = ['BinanceAdapter','BybitAdapter','OKXAdapter','CoinbaseAdapter','KrakenAdapter']
