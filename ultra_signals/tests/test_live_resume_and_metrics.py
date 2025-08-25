import asyncio, time
import pytest
from ultra_signals.live.order_exec import OrderExecutor, make_client_order_id
from ultra_signals.live.state_store import StateStore
from ultra_signals.live.metrics import Metrics
from ultra_signals.live.safety import SafetyManager


@pytest.mark.asyncio
async def test_resume_duplicate_not_replayed(tmp_path):
    store_path = tmp_path/"live_state.db"
    store = StateStore(path=str(store_path))
    existing_plan = {"ts":1, "symbol":"BTCUSDT", "side":"BUY", "price":100, "version":1}
    existing_id = make_client_order_id(existing_plan)
    store.ensure_order(existing_id)
    q = asyncio.Queue()
    metrics = Metrics()
    safety = SafetyManager(10,10,10,60,5000)
    exec_ = OrderExecutor(q, store, rate_limits={"orders_per_sec":100}, retry_cfg={"max_attempts":1,"base_delay_ms":1}, dry_run=True, safety=safety, metrics=metrics)
    await q.put(existing_plan)
    task = asyncio.create_task(exec_.run())
    await asyncio.sleep(0.1)
    exec_.stop(); task.cancel()
    assert metrics.counters.get("orders_sent",0) == 0


def test_safety_staleness_trip():
    safety = SafetyManager(10,10,10,60, data_staleness_ms=5)
    assert safety.check_data_fresh(2) is True
    assert safety.check_data_fresh(10) is False
    assert safety.state.paused and safety.state.reason == "DATA_STALENESS"


@pytest.mark.asyncio
async def test_decision_to_order_latency_metric():
    store = StateStore(path=":memory:")
    q = asyncio.Queue()
    metrics = Metrics()
    safety = SafetyManager(10,10,10,60,5000)
    exec_ = OrderExecutor(q, store, rate_limits={"orders_per_sec":100}, retry_cfg={"max_attempts":1,"base_delay_ms":1}, dry_run=True, safety=safety, metrics=metrics)
    # Force random.random() to return > 0.2 so partial path not taken
    import random as _r
    orig_rand = _r.random
    _r.random = lambda: 0.9
    try:
        plan = {"ts":1, "symbol":"BTCUSDT", "side":"BUY", "price":100, "version":1, "_decision_monotonic": time.perf_counter()-0.05}
        await q.put(plan)
        task = asyncio.create_task(exec_.run())
        await asyncio.sleep(0.1)
        exec_.stop(); task.cancel()
    finally:
        _r.random = orig_rand
    snap = metrics.latency_decision_to_order.snapshot()
    assert snap.get("count",0) == 1 and snap.get("p50",0) >= 50