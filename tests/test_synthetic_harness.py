import asyncio

from ultra_signals.lowlat.runner import synthetic_run


def test_synthetic_run():
    loop = asyncio.new_event_loop()
    try:
        metrics = loop.run_until_complete(synthetic_run(peek_n=200))
    finally:
        loop.close()

    snap = metrics.snapshot()
    assert snap["counters"]["orders_sent"] >= 0
    assert "latency_tick_to_decision" in snap
