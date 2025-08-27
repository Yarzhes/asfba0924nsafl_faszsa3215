import os
from ultra_signals.persist.db import init_db, upsert_order_pending, fetchone, update_order_after_ack, record_fill, upsert_offset, get_offset
from ultra_signals.persist.migrations import apply_migrations as run_migrations

def setup_module(m):
    if os.path.exists('live_state.db'):
        os.remove('live_state.db')
    init_db('live_state.db')
    run_migrations()


def test_journal_atomicity_and_idempotent_insert():
    order = {
        'client_order_id': 'abc123', 'venue':'binance_usdm','symbol':'BTCUSDT','side':'BUY','type':'LIMIT','qty':1.0,'price':50000.0,'reduce_only':False,'parent_id':None,'profile_id':'p1','cfg_hash':'h1'
    }
    upsert_order_pending(order)
    upsert_order_pending(order)  # idempotent ignore
    row = fetchone("SELECT status, qty FROM orders_outbox WHERE client_order_id=?", ('abc123',))
    assert row['status'] == 'PENDING'
    assert row['qty'] == 1.0


def test_update_after_ack():
    update_order_after_ack('abc123', status='ACKED', venue_order_id='X1')
    row = fetchone("SELECT status, venue_order_id FROM orders_outbox WHERE client_order_id=?", ('abc123',))
    assert row['status'] == 'ACKED'
    assert row['venue_order_id'] == 'X1'


def test_record_fill_and_position_recompute():
    record_fill({'fill_id':'f1','client_order_id':'abc123','venue':'binance_usdm','venue_order_id':'X1','symbol':'BTCUSDT','qty':0.4,'price':50010.0,'fee':0.1,'is_maker':1,'ts':1})
    record_fill({'fill_id':'f2','client_order_id':'abc123','venue':'binance_usdm','venue_order_id':'X1','symbol':'BTCUSDT','qty':0.6,'price':49990.0,'fee':0.1,'is_maker':1,'ts':2})
    pos = fetchone("SELECT qty, avg_px FROM positions WHERE symbol='BTCUSDT'")
    assert abs(pos['qty'] - 1.0) < 1e-9
    assert 49990.0 <= pos['avg_px'] <= 50010.0


def test_offsets_roundtrip():
    upsert_offset('BTCUSDT:5m', 123456, 7)
    off = get_offset('BTCUSDT:5m')
    assert off['last_ts'] == 123456
    assert off['last_seq'] == 7
