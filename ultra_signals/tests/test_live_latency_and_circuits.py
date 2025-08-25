import asyncio
import pytest
from ultra_signals.live.engine_worker import EngineWorker
from ultra_signals.live.order_exec import OrderExecutor, make_client_order_id
from ultra_signals.live.state_store import StateStore
from ultra_signals.live.safety import SafetyManager
from ultra_signals.live.metrics import Metrics
from ultra_signals.core.events import KlineEvent


@pytest.mark.asyncio
async def test_engine_latency_abstain():
    in_q = asyncio.Queue()
    out_q = asyncio.Queue(maxsize=10)
    metrics = Metrics()
    # Budget 5ms but force 20ms extra delay to trigger abstain
    eng = EngineWorker(in_q, out_q, latency_budget_ms=5, metrics=metrics, extra_delay_ms=20)
    evt = KlineEvent(timestamp=1, symbol="BTCUSDT", timeframe="5m", open=1, high=1, low=1, close=1, volume=1, closed=True)
    setattr(evt, "_ingest_monotonic", __import__("time").perf_counter())
    await in_q.put(evt)
    task = asyncio.create_task(eng.run())
    await asyncio.sleep(0.05)
    eng.stop()
    task.cancel()
    assert out_q.qsize() == 0  # abstained due to deadline


@pytest.mark.asyncio
async def test_circuit_breaker_order_error_burst():
    q = asyncio.Queue()
    store = StateStore(path=":memory:")
    safety = SafetyManager(daily_loss_limit_pct=10.0, max_consecutive_losses=10, order_error_burst_count=2, order_error_burst_window_sec=10, data_staleness_ms=5000)
    metrics = Metrics()
    # Sender that always raises
    def failing_sender(plan, client_id):
        raise RuntimeError("forced error")
    exec_ = OrderExecutor(q, store, rate_limits={"orders_per_sec": 100}, retry_cfg={"max_attempts": 1, "base_delay_ms": 1}, dry_run=False, safety=safety, metrics=metrics, order_sender=failing_sender)
    plan = {"ts":1, "symbol":"BTCUSDT", "side":"BUY", "price":100, "version":1}
    await q.put(plan)
    await q.put(plan)  # duplicate ignored, keep queue simple
    task = asyncio.create_task(exec_.run())
    await asyncio.sleep(0.2)
    exec_.stop()
    task.cancel()
    # After first (unique) order attempt error, safety should have recorded >=1; not paused yet.
    # We inject another distinct plan to trigger second error -> trip
    plan2 = {"ts":2, "symbol":"BTCUSDT", "side":"BUY", "price":100, "version":1}
    await q.put(plan2)
    task2 = asyncio.create_task(exec_.run())
    await asyncio.sleep(0.2)
    exec_.stop()
    task2.cancel()
    assert safety.state.paused is True and safety.state.reason == "ORDER_ERRORS"