"""Minimal OKX adapter skeleton (mock mode)."""
import time
from typing import Iterable, Dict, Any


class OKXAdapter:
    def __init__(self, engine, symbols=None):
        self.engine = engine
        self.symbols = symbols or []
        self._running = False

    def start(self, trade_iter: Iterable[Dict[str, Any]] = None, interval: float = 0.015, count: int = 100):
        self._running = True
        if trade_iter is None:
            trade_iter = []
            for i in range(count):
                trade_iter.append({"ts": int(time.time()), "price": 300.0 + (i%4)*0.05, "qty": 1.5 + (i%2), "side": 'buy' if i%4<2 else 'sell'})
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


__all__ = ["OKXAdapter"]
