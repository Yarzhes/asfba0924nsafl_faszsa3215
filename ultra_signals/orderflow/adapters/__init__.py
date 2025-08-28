"""Adapters for exchange websockets to feed OrderflowEngine.

Each adapter exposes a `run(engine, symbols)` coroutine or start() method in sync
mode for tests. They are skeletons meant to be filled with real websocket clients.
"""
from .binance import BinanceAdapter
from .bybit import BybitAdapter
from .okx import OKXAdapter
from .coinbase import CoinbaseAdapter
from .binance_async import BinanceAsyncAdapter
from .bybit_async import BybitAsyncAdapter
from .okx_async import OKXAsyncAdapter

__all__ = [
	"BinanceAdapter",
	"BybitAdapter",
	"OKXAdapter",
	"CoinbaseAdapter",
	"BinanceAsyncAdapter",
	"BybitAsyncAdapter",
	"OKXAsyncAdapter",
]
