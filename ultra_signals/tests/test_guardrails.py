import asyncio, time
import pytest
from ultra_signals.live.engine_worker import EngineWorker
from ultra_signals.live.metrics import Metrics
from ultra_signals.live.safety import SafetyManager
from ultra_signals.core.events import KlineEvent, BookTickerEvent


@pytest.mark.asyncio
async def test_spread_guardrail_abstains():
    in_q = asyncio.Queue()
    out_q = asyncio.Queue()
    metrics = Metrics()
    safety = SafetyManager(10,10,10,60,5000)
    eng = EngineWorker(in_q, out_q, latency_budget_ms=80, metrics=metrics, safety=safety)
    # Wide spread ~1%
    bt = BookTickerEvent(timestamp=1, symbol="BTCUSDT", b=100, B=1, a=101, A=1)
    await in_q.put(bt)
    k = KlineEvent(timestamp=2, symbol="BTCUSDT", timeframe="5m", open=1, high=1, low=1, close=1, volume=1, closed=True)
    setattr(k, "_ingest_monotonic", time.perf_counter())
    await in_q.put(k)
    task = asyncio.create_task(eng.run())
    await asyncio.sleep(0.1)
    eng.stop(); task.cancel()
    assert out_q.qsize() == 0


def test_clock_kill_switch():
    safety = SafetyManager(10,10,10,60,5000)
    safety.kill_switch("CLOCK")
    assert safety.state.paused and safety.state.reason == "CLOCK"