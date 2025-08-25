"""CLI entrypoint for Sprint 21 live trading runner."""
from __future__ import annotations
import argparse
import asyncio
from loguru import logger
from ultra_signals.core.config import load_settings
from ultra_signals.live.runner import LiveRunner


def parse_args():
    p = argparse.ArgumentParser(description="Ultra-Signals Live Runner")
    p.add_argument("--config", default="settings.yaml")
    p.add_argument("--dry-run", action="store_true", help="Force dry-run mode (no live orders)")
    p.add_argument("--latency-bench", type=int, default=0, help="Inject N synthetic closed klines to benchmark latency then exit")
    return p.parse_args()


async def _amain():
    args = parse_args()
    settings = load_settings(args.config)
    lr = LiveRunner(settings, dry_run=args.dry_run or getattr(settings.live, 'dry_run', True))
    if args.latency_bench > 0:
        # Run synthetic bench without network feeds
        from ultra_signals.core.events import KlineEvent
        lr.engine.extra_delay_ms = 0
        started = __import__("time").perf_counter()
        async def inject():
            for i in range(args.latency_bench):
                evt = KlineEvent(timestamp=started*1000+i, symbol=lr.settings.runtime.symbols[0], timeframe=lr.settings.runtime.primary_timeframe, open=1, high=1, low=1, close=1, volume=1, closed=True)
                setattr(evt, "_ingest_monotonic", __import__("time").perf_counter())
                await lr.feed_q.put(evt)
                await asyncio.sleep(0)
        await lr.start()
        await inject()
        # allow engine + executor to drain
        await asyncio.sleep(0.2)
        snap = lr.metrics.snapshot()
        logger.info(f"[LatencyBench] snapshot={snap}")
        await lr.stop()
        return
    else:
        try:
            await lr.start()
            # Keep process running until Ctrl+C
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            logger.warning("Ctrl+C received; shutting down")
        finally:
            await lr.stop()


def run():
    logger.remove()
    logger.add(lambda m: print(m, end=""), level="INFO")
    asyncio.run(_amain())


if __name__ == "__main__":  # pragma: no cover
    run()
