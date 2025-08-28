import time
from ultra_signals.features.vpin import VPINEngine


def test_vpin_bucket_finalize_basic():
    eng = VPINEngine(V_bucket=1000.0, K_buckets=5)
    # create synthetic trades: eight buys of 250@2 => each trade notional=500, total=4000 -> should finalize 4 buckets at V_bucket=1000
    ts = int(time.time() * 1000)
    trades = []
    for i in range(8):
        trades.append((ts + i, 2.0, 250.0, False))
    for tr in trades:
        eng.ingest_trade(tr)
    # after ingestion, there should be 4 buckets finalized
    assert eng.get_buckets_summary()['total_buckets'] == 4
    latest = eng.get_latest_vpin()
    assert 'vpin' in latest


def test_tick_rule_fallback():
    eng = VPINEngine(V_bucket=10000.0, K_buckets=3)
    ts = int(time.time() * 1000)
    # no is_buyer_maker info, but provide book_top
    book = {'bid': 99.0, 'ask': 101.0}
    eng.ingest_trade((ts, 101.0, 100.0, None), book_top=book)  # price >= mid => BUY
    eng.ingest_trade((ts+1, 98.0, 100.0, None), book_top=book)  # price < mid => SELL
    # ensure current bucket captured both sides
    p = eng.finalize_partial()
    assert p['class_error_est'] == 0
