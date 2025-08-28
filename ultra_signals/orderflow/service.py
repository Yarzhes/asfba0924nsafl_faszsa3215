"""Orderflow service that wires the calculator to persistence and a simulated feed.

Provides:
 - SimulatedFeed: simple deterministic trade/book generator for demos and tests
 - OrderflowService: periodically ingests feed into calculator and persists FeatureView records

The module is intentionally dependency-light to be import-safe in tests.
"""

from __future__ import annotations

import random
import time
from typing import List, Tuple, Optional

from .engine import OrderflowCalculator
from .persistence import FeatureViewWriter


class SimulatedFeed:
    """Generate deterministic trades and book snapshots for a single symbol.

    Trades have fields: ts, size, price, side
    Books are lists of (price, size) tuples for bids and asks.
    """

    def __init__(self, symbol: str = "SIM", base_price: float = 100.0, seed: Optional[int] = None):
        self.symbol = symbol
        self.base_price = base_price
        self._rng = random.Random(seed)

    def generate_trade(self, ts: Optional[float] = None) -> dict:
        ts = ts or time.time()
        # small random walk around base_price
        px = self.base_price + self._rng.uniform(-0.5, 0.5)
        side = self._rng.choice(["buy", "sell"])
        size = float(round(self._rng.expovariate(1 / 50.0), 6))  # mean size ~50
        return {"ts": ts, "size": size, "price": px, "side": side}

    def generate_book(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        # create a small book around base_price
        bids = []
        asks = []
        for i in range(5):
            bids.append((self.base_price - 0.1 * i, float(round(100 * self._rng.random(), 3))))
            asks.append((self.base_price + 0.1 * i + 0.1, float(round(100 * self._rng.random(), 3))))
        return bids, asks


class OrderflowService:
    """Service that wires stream -> calculator -> writer.

    Usage:
      svc = OrderflowService(calc, writer, feed, interval_s=1.0)
      svc.run_loop(duration_s=10)

    For tests use `run_once()` which executes a single ingest+persist cycle.
    """

    def __init__(self, calculator: OrderflowCalculator, writer: FeatureViewWriter, feed: SimulatedFeed, interval_s: float = 1.0, symbol: str = "SIM") -> None:
        self.calculator = calculator
        self.writer = writer
        self.feed = feed
        self.interval_s = interval_s
        self.symbol = symbol
        self._running = False
        # local RNG for the service
        self._rng = random.Random()

    def run_once(self, ts: Optional[float] = None) -> dict:
        """Run a single ingest -> compute -> persist cycle and return the record."""
        ts = ts or time.time()
        bids, asks = self.feed.generate_book()
        self.calculator.ingest_book_snapshot(bids, asks)

        last_trade = None
        for _ in range(self._rng_trade_count()):
            t = self.feed.generate_trade(ts=ts)
            self.calculator.ingest_trade(t)
            last_trade = t

        metrics = self.calculator.compute_current(now=ts)
        record = {
            "ts": int(ts),
            "symbol": self.symbol,
            "price": float(last_trade["price"]) if last_trade is not None else None,
            "of_micro_score": self._fuse_score(metrics),
            "components": metrics,
        }

        self.writer.write_record(record)
        return record

    def _rng_trade_count(self) -> int:
        """Return a small random number of trades per interval (0-3)."""
        return self._rng.randint(0, 3)

    def _fuse_score(self, metrics: dict) -> float:
        """Simple fusion: weighted sum of a few normalized components (demo only)."""
        cvd = float(metrics.get("cvd_pct") or 0.0)
        ob = float(metrics.get("ob_imbalance_top1") or 0.0)
        burst = float(metrics.get("tape_burst_flag") or 0.0)
        score = 0.6 * cvd + 0.3 * ob - 0.5 * burst
        return float(score)

    def run_loop(self, duration_s: float = 10.0) -> None:
        """Run the ingest/persist loop for duration_s seconds (blocking)."""
        self._running = True
        end = time.time() + duration_s
        while self._running and time.time() < end:
            ts = time.time()
            rec = self.run_once(ts=ts)
            print(f"Wrote record: ts={rec['ts']} symbol={rec['symbol']} score={rec['of_micro_score']:.4f} price={rec['price']}")
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._running = False


# small CLI demo when run as a script
if __name__ == "__main__":
    import argparse
    import tempfile

    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()

    # demo uses a temp sqlite file in current directory
    dbpath = "orderflow_demo.db"
    writer = FeatureViewWriter(sqlite_path=dbpath)
    calc = OrderflowCalculator(vps_window_s=5, cvd_window_s=10)
    feed = SimulatedFeed(symbol="DEMO", base_price=100.0, seed=42)
    svc = OrderflowService(calc, writer, feed, interval_s=args.interval, symbol="DEMO")
    try:
        svc.run_loop(duration_s=args.duration)
    finally:
        writer.close()
