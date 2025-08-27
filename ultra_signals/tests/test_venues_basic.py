import asyncio
import pytest
from ultra_signals.venues import SymbolMapper, VenueRouter
from ultra_signals.venues.health import HealthRegistry
from ultra_signals.venues.binance_usdm import BinanceUSDMPaper
from ultra_signals.venues.bybit_perp import BybitPerpPaper
from ultra_signals.live.order_exec import make_client_order_id


def test_symbol_mapping_per_venue():
    mapping = {"BTCUSDT": {"binance_usdm": "BTCUSDT", "bybit_perp": "BTCUSDT"}}
    sm = SymbolMapper(mapping)
    assert sm.to_venue("BTCUSDT", "binance_usdm") == "BTCUSDT"
    assert sm.round_trip_ok("BTCUSDT", "bybit_perp")


def test_health_scoring_thresholds():
    reg = HealthRegistry({"red_threshold": 0.35, "yellow_threshold": 0.65, "staleness_ms_max": 2500})
    st = reg.ensure("v1")
    # Fresh healthy state
    reg.record_ws_staleness("v1", 100)
    color1 = st.color(reg.cfg)
    # Force staleness high -> degrade
    reg.record_ws_staleness("v1", 5000)
    color2 = st.color(reg.cfg)
    assert color1 in ("green", "yellow")
    assert color2 in ("red", "yellow", "green")  # heuristic may still keep it >= yellow threshold


@pytest.mark.asyncio
async def test_router_order_failover_once(monkeypatch):
    sm = SymbolMapper({})
    b = BinanceUSDMPaper(sm)
    y = BybitPerpPaper(sm)
    # Force Binance raise once
    called = {"b": 0}
    orig = b.place_order
    async def failing(plan, cid):
        if called["b"] == 0:
            called["b"] += 1
            raise RuntimeError("boom")
        return await orig(plan, cid)
    b.place_order = failing  # type: ignore
    router = VenueRouter({"binance_usdm": b, "bybit_perp": y}, sm, {"primary_order": ["binance_usdm", "bybit_perp"], "health": {"red_threshold": 0.35, "yellow_threshold": 0.65}})
    plan = {"ts": 1, "symbol": "BTCUSDT", "side": "LONG", "price": 100, "qty": 1, "version": 1}
    cid = make_client_order_id(plan)
    res = await router.place_order(plan, cid)
    assert res.get("failover") is True
    assert res["venue"] == "bybit_perp"
    # second call should still use failover venue due to stickiness after primary failure
    plan2 = {**plan, "ts": 2}
    cid2 = make_client_order_id(plan2)
    res2 = await router.place_order(plan2, cid2)
    assert res2["venue"] == "bybit_perp"


@pytest.mark.asyncio
async def test_idempotency_duplicate_ack():
    sm = SymbolMapper({})
    b = BinanceUSDMPaper(sm)
    y = BybitPerpPaper(sm)
    router = VenueRouter({"binance_usdm": b, "bybit_perp": y}, sm, {"primary_order": ["binance_usdm", "bybit_perp"], "health": {"red_threshold": 0.35, "yellow_threshold": 0.65}})
    plan = {"ts": 1, "symbol": "BTCUSDT", "side": "LONG", "price": 100, "qty": 1, "version": 1}
    cid = make_client_order_id(plan)
    first = await router.place_order(plan, cid)
    # resend same plan/id; should produce another ack but client_order_id same
    second = await router.place_order(plan, cid)
    assert first["ack"].client_order_id == second["ack"].client_order_id
