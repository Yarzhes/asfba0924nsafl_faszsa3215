"""CLI utility to exercise VenueRouter health & failover (Sprint 23)."""
from __future__ import annotations
import argparse
import asyncio
import time
from loguru import logger
from ultra_signals.core.config import load_settings
from ultra_signals.venues import VenueRouter, SymbolMapper
from ultra_signals.venues.binance_usdm import BinanceUSDMPaper
from ultra_signals.venues.bybit_perp import BybitPerpPaper
from ultra_signals.live.order_exec import make_client_order_id


def parse_args():
    p = argparse.ArgumentParser(description="Venue health & routing demo")
    p.add_argument("--config", default="settings.yaml")
    p.add_argument("--symbols", default="BTCUSDT")
    p.add_argument("--timeframes", default="5m")
    p.add_argument("--duration", default="60s")
    p.add_argument("--spam-orders", default="false")
    return p.parse_args()


async def main():
    args = parse_args()
    settings = load_settings(args.config)
    dur_s = int(args.duration.rstrip("s"))
    symbols = args.symbols.split(",")
    tfs = args.timeframes.split(",")
    venues_cfg = getattr(settings, "venues", {}) if hasattr(settings, "venues") else {}
    mapper = SymbolMapper((venues_cfg or {}).get("symbol_map", {}))
    # Instantiate paper adapters (no network)
    adapters = {
        "binance_usdm": BinanceUSDMPaper(mapper),
        "bybit_perp": BybitPerpPaper(mapper),
    }
    router = VenueRouter(adapters, mapper, venues_cfg)
    start = time.time()
    next_health = 0
    order_toggle = 0
    while time.time() - start < dur_s:
        now = time.time()
        # simulate ws staleness drift
        for vid in adapters.keys():
            router.health.record_ws_staleness(vid, (now - start) * 10)  # synthetic
        if now >= next_health:
            snap = router.snapshot()
            logger.info(f"[venue_check] snapshot={snap}")
            next_health = now + 5
        # Optionally spam synthetic orders
        if args.spam_orders.lower() == "true":
            for sym in symbols:
                plan = {"ts": int(now*1000), "symbol": sym, "side": "LONG" if order_toggle % 2 == 0 else "SHORT", "price": 100, "qty": 1, "version": 1}
                cid = make_client_order_id(plan)
                try:
                    res = await router.place_order(plan, cid)
                    logger.info(f"order={res}")
                except Exception as e:
                    logger.error(f"order error {e}")
                order_toggle += 1
        await asyncio.sleep(1)


def run():  # pragma: no cover
    asyncio.run(main())

if __name__ == "__main__":  # pragma: no cover
    run()
