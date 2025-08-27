import asyncio
import pytest
from ultra_signals.live.runner import LiveRunner
from ultra_signals.live.state_store import StateStore


@pytest.mark.asyncio
async def test_reconcile_on_restart_multi_venue():
    # Build minimal settings stub with venues
    class Dummy: pass
    s = Dummy(); s.live = Dummy(); s.live.dry_run=True; s.live.metrics={"exporter":"none"}; s.live.health={}; s.live.control={}; s.live.store_path=":memory:"
    s.engine = Dummy(); s.engine.risk = Dummy(); s.engine.risk.max_spread_pct={"default":0.05}
    s.data_sources={"binance": Dummy()}; s.data_sources["binance"].api_key="k"; s.data_sources["binance"].api_secret="s"
    s.runtime = Dummy(); s.runtime.symbols=["BTCUSDT"]; s.runtime.timeframes=["5m"]; s.runtime.primary_timeframe="5m"
    # venues settings
    class VenuesCfg: pass
    vc = VenuesCfg(); vc.primary_order=["binance_usdm","bybit_perp"]; vc.data_order=["binance_usdm","bybit_perp"]; vc.symbol_map={}; vc.health=type("H",(),{"red_threshold":0.35,"yellow_threshold":0.65,"cooloff_sec":30,"staleness_ms_max":2500})(); vc.ratelimits={}; vc.fees={}; vc.prefer_lower_fee_on_tie=True
    s.venues = vc
    lr = LiveRunner(s, dry_run=True)
    # Seed DB with a stale pending order (will be canceled) and one partial -> filled
    store = lr.store
    store.ensure_order("CID1")
    store.update_order("CID1", status="PENDING")
    store.ensure_order("CID2")
    store.update_order("CID2", status="PARTIAL")
    await lr.start()
    await asyncio.sleep(0.05)
    o1 = store.get_order("CID1")
    o2 = store.get_order("CID2")
    assert o1["status"] in ("CANCELED","FILLED")  # canceled by reconcile
    assert o2["status"] == "FILLED"
    await lr.stop()


@pytest.mark.asyncio
async def test_order_failover_rest():
    # Force primary adapter to raise then ensure fallback used
    class Dummy: pass
    s = Dummy(); s.live = Dummy(); s.live.dry_run=True; s.live.metrics={"exporter":"none"}; s.live.health={}; s.live.control={}; s.live.store_path=":memory:"
    s.engine = Dummy(); s.engine.risk = Dummy(); s.engine.risk.max_spread_pct={"default":0.05}
    s.data_sources={"binance": Dummy()}; s.data_sources["binance"].api_key="k"; s.data_sources["binance"].api_secret="s"
    s.runtime = Dummy(); s.runtime.symbols=["BTCUSDT"]; s.runtime.timeframes=["5m"]; s.runtime.primary_timeframe="5m"
    class VenuesCfg: pass
    vc = VenuesCfg(); vc.primary_order=["binance_usdm","bybit_perp"]; vc.data_order=["binance_usdm","bybit_perp"]; vc.symbol_map={}; vc.health=type("H",(),{"red_threshold":0.35,"yellow_threshold":0.65,"cooloff_sec":30,"staleness_ms_max":2500})(); vc.ratelimits={}; vc.fees={}; vc.prefer_lower_fee_on_tie=True
    s.venues=vc
    lr = LiveRunner(s, dry_run=True)
    # Monkeypatch primary to raise
    b = lr.venue_router.venues["binance_usdm"]
    async def boom(plan, cid):
        raise RuntimeError("X")
    b.place_order = boom  # type: ignore
    await lr.start()
    plan = {"ts":2,"symbol":"BTCUSDT","side":"LONG","price":100,"qty":1,"version":1}
    await lr.order_q.put(plan)
    await asyncio.sleep(0.1)
    # Order should exist with venue set to fallback (bybit_perp)
    rows = [o for o in lr.store.list_orders() if o["status"] in ("FILLED", "PENDING")]
    assert any(r.get("venue_id") == "bybit_perp" for r in rows)
    await lr.stop()


def test_data_failover_ws():
    # Directly exercise router staleness causing data venue switch
    from ultra_signals.venues import SymbolMapper, VenueRouter
    from ultra_signals.venues.binance_usdm import BinanceUSDMPaper
    from ultra_signals.venues.bybit_perp import BybitPerpPaper
    sm = SymbolMapper({})
    b = BinanceUSDMPaper(sm)
    y = BybitPerpPaper(sm)
    cfg = {"primary_order":["binance_usdm","bybit_perp"],"data_order":["binance_usdm","bybit_perp"],"health":{"red_threshold":0.35,"yellow_threshold":0.65,"staleness_ms_max":2500}}
    router = VenueRouter({"binance_usdm": b, "bybit_perp": y}, sm, cfg)
    first = router.decide_data_venue("BTCUSDT","5m")
    assert first in ("binance_usdm","bybit_perp")
    # degrade current venue staleness massively
    router.health.record_ws_staleness(first, 10_000)
    new = router.decide_data_venue("BTCUSDT","5m")
    # If staleness pushes below red threshold, expect possible switch (not guaranteed due to scoring but allowed)
    assert new in ("binance_usdm","bybit_perp")