import os, tempfile, json, asyncio, time
from ultra_signals.persist.db import init_db, execute, fetchall
from ultra_signals.persist.migrations import apply_migrations
from ultra_signals.core.alerts import publish_alert, recent_alerts


def setup_db(tmp_path):
    os.environ['ULTRA_SIGNALS_DB_PATH'] = str(tmp_path/'test.db')
    init_db()
    apply_migrations()


def test_alert_persist_and_stream(tmp_path):
    setup_db(tmp_path)
    publish_alert('TEST_EVENT','Unit test alert', severity='WARN', meta={'x':1})
    alerts = recent_alerts(5)
    assert any(a['type']=='TEST_EVENT' for a in alerts)


def test_equity_curve_append(tmp_path):
    setup_db(tmp_path)
    ts = int(time.time()*1000)
    execute("INSERT INTO equity_curve(ts,equity,drawdown) VALUES(?,?,?)", (ts, 1000.0, 0.0))
    rows = fetchall("SELECT * FROM equity_curve")
    assert rows and rows[0]['equity'] == 1000.0


def test_api_status_snapshot(tmp_path):
    # Directly test snapshot util from dashboard via import (compute_snapshot)
    setup_db(tmp_path)
    # Insert dummy position & order
    execute("INSERT INTO positions(symbol, qty, avg_px, realized_pnl, updated_ts, venue, hedge) VALUES('BTCUSDT',1.0,25000,0.0,strftime('%s','now')*1000,'binance',0)")
    execute("INSERT INTO orders_outbox(client_order_id, venue, symbol, side, type, qty, price, reduce_only, parent_id, status, venue_order_id, last_error, retries, created_ts, updated_ts, profile_id, cfg_hash) VALUES('cid1','binance','BTCUSDT','BUY','LIMIT',1,25000,0,NULL,'NEW',NULL,NULL,0,strftime('%s','now')*1000,strftime('%s','now')*1000,NULL,NULL)")
    from ultra_signals.apps.dashboard import compute_snapshot
    snap = compute_snapshot()
    assert isinstance(snap, dict)
    assert any(p['symbol']=='BTCUSDT' for p in snap['positions'])
    assert snap['orders']
