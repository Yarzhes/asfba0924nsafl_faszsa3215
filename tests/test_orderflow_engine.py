import time
from ultra_signals.orderflow.engine import OrderflowEngine


def test_cvd_basic():
    eng = OrderflowEngine({"cvd_window": 60})
    now = time.time()
    eng.ingest_trade(now-1, 100.0, 10, "buy", aggressor=True)
    eng.ingest_trade(now-1, 100.5, 5, "sell", aggressor=True)
    cvd = eng.get_cvd()
    assert abs(cvd["cvd_abs"] - 5.0) < 1e-6
    assert cvd["cvd_pct"] > 0


def test_orderbook_imbalance():
    eng = OrderflowEngine()
    eng.ingest_orderbook_snapshot(bids=[(100, 50), (99.5, 20)], asks=[(100.5, 10), (101, 5)])
    imb, bid_sum, ask_sum = eng.get_orderbook_imbalance(topN=2)
    assert bid_sum == 70
    assert ask_sum == 15
    assert imb > 0


def test_tape_burst():
    eng = OrderflowEngine({"tape_window": 2, "tape_z_window": 10})
    now = time.time()
    # low baseline
    for i in range(5):
        eng.ingest_trade(now - 20 + i, 100, 1, "buy", aggressor=True)
    # burst
    for i in range(20):
        eng.ingest_trade(now - 1 + i*0.01, 100, 50, "sell", aggressor=True)
    m = eng.get_tape_metrics(now_ts=now)
    assert m["vps"] > 0


def test_footprint_levels():
    eng = OrderflowEngine({"footprint_min_volume": 10})
    now = time.time()
    eng.ingest_trade(now, 100.0, 5, "buy", aggressor=True)
    eng.ingest_trade(now, 100.0, 7, "sell", aggressor=True)
    levels = eng.detect_footprint_levels()
    assert levels and levels[0][0] == 100.0
