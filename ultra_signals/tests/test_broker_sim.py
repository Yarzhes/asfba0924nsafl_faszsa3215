import math
from ultra_signals.sim.broker import BrokerSim, Order, map_side
from ultra_signals.sim.orderbook import SyntheticOrderBook

BASIC_CFG = {
    'venue_defaults': {'maker_fee_bps': -1.0, 'taker_fee_bps': 4.0},
    'venues': { 'SIM': { 'latency_ms': { 'submit': {'fixed':1}, 'match': {'fixed':1} }, 'slippage': {'impact_factor': 0.5, 'jitter_bps': {'dist':'normal','mean':0,'std':0.0}} } },
    'orderbook': {'levels': 5},
    'policies': {'partial_fill_min_ratio': 0.05}
}

def build_sim():
    ob = SyntheticOrderBook('TEST', levels=5, seed=123)
    ob.rebuild_from_bar({'close': 100.0, 'high': 100.5, 'low': 99.5})
    sim = BrokerSim(BASIC_CFG, ob, rng_seed=7, venue='SIM')
    return sim


def test_market_sweep_vwap():
    sim = build_sim()
    o = Order(id='1', symbol='TEST', side='BUY', type='MARKET', qty=1.0)
    fills = sim.submit_order(o)
    assert fills, 'Expected fill'
    vwap = sum(f.price*f.qty for f in fills)/sum(f.qty for f in fills)
    assert vwap>0


def test_limit_queue_fill_order():
    sim = build_sim()
    # place two limits same price
    obp = sim.orderbook.best_bid()
    o1 = Order(id='A', symbol='TEST', side='BUY', type='LIMIT', qty=1.0, price=obp*0.999)
    o2 = Order(id='B', symbol='TEST', side='BUY', type='LIMIT', qty=1.0, price=obp*0.999)
    sim.submit_order(o1); sim.submit_order(o2)
    # advance time enough for queue consumption
    sim.advance_time(500)
    filled_ids = {f.order_id for f in sim.fills}
    assert 'A' in filled_ids, 'First order should fill before second'


def test_post_only_reject():
    sim = build_sim()
    # post only crossing (buy at or above ask)
    ask = sim.orderbook.best_ask()
    o = Order(id='PO', symbol='TEST', side='BUY', type='POST_ONLY', qty=1.0, price=ask)
    fills = sim.submit_order(o)
    assert not fills, 'Post only crossing should reject (no fills)'


def test_fees_signs():
    sim = build_sim()
    o = Order(id='MKT', symbol='TEST', side='BUY', type='MARKET', qty=0.5)
    fills = sim.submit_order(o)
    assert fills[0].fee_bps == 4.0
    # maker
    bid = sim.orderbook.best_bid()*0.999
    lm = Order(id='L', symbol='TEST', side='BUY', type='LIMIT', qty=0.5, price=bid)
    sim.submit_order(lm)
    sim.advance_time(400)
    maker_fill = [f for f in sim.fills if f.order_id=='L'][-1]
    assert maker_fill.fee_bps == -1.0


def test_rng_seed_repro():
    sim1 = build_sim(); sim2 = build_sim()
    o1 = Order(id='R1', symbol='TEST', side='BUY', type='MARKET', qty=1.0)
    o2 = Order(id='R2', symbol='TEST', side='BUY', type='MARKET', qty=1.0)
    f1 = sim1.submit_order(o1); f2 = sim2.submit_order(o2)
    assert f1[0].price == f2[0].price


def test_cancel_race():
    sim = build_sim()
    price = sim.orderbook.best_bid()*0.999
    o = Order(id='C1', symbol='TEST', side='BUY', type='LIMIT', qty=2.0, price=price)
    sim.submit_order(o)
    # advance a bit but attempt cancel mid-way
    sim.advance_time(50)
    partially_filled = any(f.order_id=='C1' for f in sim.fills)
    sim.cancel_order('C1')
    # after cancel no further fills should appear
    fill_count = len([f for f in sim.fills if f.order_id=='C1'])
    sim.advance_time(200)
    fill_count_after = len([f for f in sim.fills if f.order_id=='C1'])
    assert fill_count_after == fill_count, 'Cancel race allowed extra fills'
