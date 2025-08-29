import asyncio
import time
from ultra_signals.live.engine_worker import EngineWorker
from ultra_signals.live.metrics import Metrics
from ultra_signals.core.events import KlineEvent, BookTickerEvent

import pytest

@pytest.mark.asyncio
async def test_closed_kline_emits_signal_metrics():
    in_q = asyncio.Queue()
    out_q = asyncio.Queue()
    metrics = Metrics()
    eng = EngineWorker(in_q, out_q, latency_budget_ms=250, metrics=metrics, safety=None)

    async def run_engine():
        await eng.run()

    task = asyncio.create_task(run_engine())

    # Provide a recent book ticker so spread guard passes
    bt = BookTickerEvent(timestamp=int(time.time()*1000), symbol='BTCUSDT', b=50000.0, B=1.0, a=50000.5, A=1.0)
    await in_q.put(bt)
    # Closed kline event (close>open triggers LONG in placeholder strategy)
    kl = KlineEvent(timestamp=int(time.time()*1000), symbol='BTCUSDT', timeframe='1m', open=100.0, high=101.0, low=99.5, close=100.5, volume=123.0, closed=True)
    await in_q.put(kl)

    # Allow processing
    await asyncio.sleep(0.2)

    # Stop engine
    eng.stop()
    task.cancel()
    try:
        await task
    except Exception:
        pass

    snap = metrics.snapshot()
    assert snap['counters']['signals_candidates'] >= 1, snap
    # allowed or blocked will depend on filters; sum should equal candidates
    allowed = snap['counters']['signals_allowed']
    blocked = snap['counters']['signals_blocked']
    assert allowed + blocked == snap['counters']['signals_candidates'], snap
