import time
import pytest
from ultra_signals.venues import SymbolMapper, VenueRouter
from ultra_signals.venues.binance_usdm import BinanceUSDMPaper
from ultra_signals.venues.bybit_perp import BybitPerpPaper


def test_all_red_detection():
    sm = SymbolMapper({})
    b = BinanceUSDMPaper(sm)
    y = BybitPerpPaper(sm)
    router = VenueRouter({"binance_usdm": b, "bybit_perp": y}, sm, {"primary_order": ["binance_usdm","bybit_perp"], "data_order": ["binance_usdm","bybit_perp"], "health": {"red_threshold": 0.99, "yellow_threshold": 0.995}})
    # With high red threshold initial scores ~1.0 still maybe >= threshold; raise staleness massively to degrade
    router.health.record_ws_staleness("binance_usdm", 10_000)
    router.health.record_ws_staleness("bybit_perp", 10_000)
    assert router.all_order_venues_red() or router.all_data_venues_red()


def test_router_overhead():
    sm = SymbolMapper({})
    b = BinanceUSDMPaper(sm)
    y = BybitPerpPaper(sm)
    router = VenueRouter({"binance_usdm": b, "bybit_perp": y}, sm, {"primary_order": ["binance_usdm","bybit_perp"], "data_order": ["binance_usdm","bybit_perp"], "health": {"red_threshold": 0.10, "yellow_threshold": 0.5}})
    N = 500
    start = time.perf_counter()
    for i in range(N):
        router.decide_order_venue("BTCUSDT", "LONG")
    dur_ms = (time.perf_counter() - start)*1000
    p99_est = dur_ms / N  # rough since constant cost
    assert p99_est < 2.0, f"Router overhead too high ~{p99_est:.3f}ms"
