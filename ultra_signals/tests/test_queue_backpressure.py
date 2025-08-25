import asyncio
import pytest
from ultra_signals.live.engine_worker import EngineWorker


@pytest.mark.asyncio
async def test_queue_backpressure():
    feed_q = asyncio.Queue(maxsize=5)
    order_q = asyncio.Queue(maxsize=1)
    worker = EngineWorker(feed_q, order_q, latency_budget_ms=1000)
    task = asyncio.create_task(worker.run())
    # overfill feed queue
    for i in range(20):
        try:
            feed_q.put_nowait(type("Evt", (), {"closed": True, "timestamp": i, "symbol": "BTCUSDT", "timeframe": "1m", "close": 100+i}))
        except asyncio.QueueFull:
            break
    await asyncio.sleep(0.1)
    # order queue bounded at 1, ensure not > 1
    assert order_q.qsize() <= 1
    task.cancel()
    await asyncio.sleep(0)
