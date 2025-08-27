import pytest
from ultra_signals.venues import SymbolMapper, VenueRouter
from ultra_signals.venues.binance_usdm import BinanceUSDMPaper
from ultra_signals.venues.bybit_perp import BybitPerpPaper
from ultra_signals.live.order_exec import make_client_order_id


def test_colocation_bias_prefers_data_venue_when_close():
    sm = SymbolMapper({})
    b = BinanceUSDMPaper(sm)
    y = BybitPerpPaper(sm)
    cfg = {"primary_order":["binance_usdm","bybit_perp"],"data_order":["binance_usdm","bybit_perp"],"health":{"red_threshold":0.35,"yellow_threshold":0.65},"colocation_bias_score_diff":0.2}
    router = VenueRouter({"binance_usdm": b, "bybit_perp": y}, sm, cfg)
    # Pick data venue first
    data_vid = router.decide_data_venue("BTCUSDT","5m")
    order_vid = router.decide_order_venue("BTCUSDT","LONG")
    assert order_vid == data_vid  # bias keeps them aligned initially


def test_reduce_only_rejects_increasing_position():
    sm = SymbolMapper({})
    b = BinanceUSDMPaper(sm)
    cfg = {"primary_order":["binance_usdm"],"data_order":["binance_usdm"],"health":{}}
    router = VenueRouter({"binance_usdm": b}, sm, cfg)
    plan = {"ts":1,"symbol":"BTCUSDT","side":"LONG","price":100,"qty":1,"version":1}
    cid = make_client_order_id(plan)
    # Normal fill
    import asyncio
    async def place():
        return await b.place_order(plan, cid)
    ack = asyncio.run(place())
    assert ack.status == "FILLED"
    # Reduce-only LONG should reject since it would increase
    plan2 = {**plan, "ts":2, "reduce_only":True}
    cid2 = make_client_order_id(plan2)
    async def place2():
        return await b.place_order(plan2, cid2)
    ack2 = asyncio.run(place2())
    assert ack2.status == "REJECTED"