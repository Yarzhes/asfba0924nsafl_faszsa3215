"""Minimal Binance adapter skeleton for OrderflowEngine.

Provides a mock `start` that emits pre-canned trades or forwards a provided
iterator. Replace with real websocket client in production.
"""
import time
from typing import Iterable, Dict, Any


class BinanceAdapter:
    def __init__(self, engine, symbols=None):
        self.engine = engine
        self.symbols = symbols or []
        self._running = False

    def start(self, trade_iter: Iterable[Dict[str, Any]] = None, interval: float = 0.01, count: int = 100):
        """Run in synchronous mock mode: iterate trade_iter and feed engine.ingest_trade."""
        self._running = True
        if trade_iter is None:
            # produce synthetic trades for tests
            trade_iter = []
            for i in range(count):
                trade_iter.append({"ts": int(time.time()), "price": 100.0 + (i%5)*0.1, "qty": 1 + (i%3), "side": 'buy' if i%2==0 else 'sell'})
        for t in trade_iter:
            if not self._running:
                break
            try:
                self.engine.ingest_trade(int(t.get('ts') or time.time()), float(t.get('price')), float(t.get('qty')), t.get('side'), aggressor=True)
            except Exception:
                pass
            time.sleep(interval)

    def stop(self):
        self._running = False


__all__ = ["BinanceAdapter"]
