import time

from ultra_signals.orderflow.engine import (
    compute_cvd,
    compute_ob_imbalance,
    compute_tape_metrics,
    detect_vps_burst,
    OrderflowCalculator,
)


def test_cvd_simple():
    trades = [
        {"size": 10.0, "side": "buy"},
        {"size": 5.0, "side": "sell"},
        {"size": 2.5, "side": "buy"},
    ]
    r = compute_cvd(trades)
    # cvd = 10 - 5 + 2.5 = 7.5
    assert abs(r["cvd_abs"] - 7.5) < 1e-9
    assert abs(r["total_volume"] - 17.5) < 1e-9
    assert abs(r["cvd_pct"] - (7.5 / 17.5)) < 1e-9


def test_ob_imbalance():
    bids = [(100.0, 5.0), (99.5, 3.0), (99.0, 2.0)]
    asks = [(100.5, 1.0), (101.0, 1.0), (101.5, 1.0)]
    imbalance_top1 = compute_ob_imbalance(bids, asks, top_n=1)
    # top1 bid=5, ask=1 -> (5-1)/(5+1)=4/6=0.666...
    assert abs(imbalance_top1 - (4.0 / 6.0)) < 1e-9
    imbalance_top3 = compute_ob_imbalance(bids, asks, top_n=3)
    bid_sum = 5.0 + 3.0 + 2.0
    ask_sum = 1.0 + 1.0 + 1.0
    assert abs(imbalance_top3 - ((bid_sum - ask_sum) / (bid_sum + ask_sum))) < 1e-9


def test_tape_metrics_and_burst():
    now = time.time()
    # create a history of vps values
    history = [100.0, 110.0, 95.0, 105.0]
    # current vps is high
    current_trades = [
        {"ts": now - 1, "size": 500.0, "price": 10.0},
        {"ts": now - 0.5, "size": 400.0, "price": 10.0},
    ]
    tape = compute_tape_metrics(current_trades, window_s=2.0)
    # total size = 900, vps = 450
    assert abs(tape["tape_vps"] - 450.0) < 1e-6
    # detect burst comparing to history
    assert detect_vps_burst(tape["tape_vps"], history, sigma=2.0) is True


def test_orderflow_calculator_integration():
    calc = OrderflowCalculator(vps_window_s=5, cvd_window_s=10)
    now = time.time()
    # ingest some book snapshot
    bids = [(100.0, 10.0), (99.5, 5.0), (99.0, 2.0)]
    asks = [(100.5, 1.0), (101.0, 1.0), (101.5, 1.0)]
    calc.ingest_book_snapshot(bids, asks)
    # ingest trades
    calc.ingest_trade({"ts": now - 1, "size": 50.0, "price": 100.1, "side": "buy"})
    calc.ingest_trade({"ts": now - 0.5, "size": 20.0, "price": 100.2, "side": "sell"})
    metrics = calc.compute_current(now=now)
    assert "cvd_abs" in metrics
    assert "ob_imbalance_top1" in metrics
    assert "tape_vps" in metrics

