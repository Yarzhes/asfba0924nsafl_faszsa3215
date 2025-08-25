import math
from ultra_signals.engine.orderflow import OrderFlowAnalyzer

def test_cvd_positive():
    trades = [
        {"price":100,"qty":2,"side":"buy"},
        {"price":101,"qty":1.5,"side":"buy"},
        {"price":100.5,"qty":1.0,"side":"sell"},
    ]
    cvd, chg = OrderFlowAnalyzer.compute_cvd(trades)
    assert round(cvd,2) == 2.5
    assert chg is None  # no prev


def test_liquidation_cluster_detection():
    liqs = [
        {"side":"long","notional":200_000},
        {"side":"short","notional":150_000},
        {"side":"short","notional":900_000},
    ]
    res = OrderFlowAnalyzer.detect_liquidation_clusters(liqs, notional_threshold=500_000)
    assert res["dominant"] == "short"
    assert res["short_notional"] > res["long_notional"]
    assert res["impulse"] >= 1.0


def test_liquidity_sweep_detection():
    ob = {
        "bids": [[100,5],[99.5,4],[99,3]],
        "asks": [[100.5,1],[101,1],[101.5,1]]
    }
    res = OrderFlowAnalyzer.detect_liquidity_sweep(ob, imbalance_threshold=2.0)
    assert res["sweep_side"] in ("ask","bid")
    assert res["imbalance"] >= 2.0
