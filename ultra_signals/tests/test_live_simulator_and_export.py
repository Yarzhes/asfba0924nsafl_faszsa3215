import asyncio, time, csv, os, random
import pytest
from ultra_signals.live.order_exec import OrderExecutor, make_client_order_id
from ultra_signals.live.state_store import StateStore
from ultra_signals.live.metrics import Metrics
from ultra_signals.live.safety import SafetyManager


@pytest.mark.asyncio
async def test_simulator_reject_and_slip(tmp_path):
    store = StateStore(path=":memory:")
    q = asyncio.Queue()
    metrics = Metrics()
    safety = SafetyManager(10,10,10,60,5000)
    # Force rejection by monkeypatching random.random sequence
    seq = [0.0, 0.5]  # first call -> reject (0.0 < reject_prob), second for part_prob skip
    def fake_random():
        return seq.pop(0) if seq else 0.9
    random_orig = random.random
    random.random = fake_random
    try:
        exec_ = OrderExecutor(q, store, rate_limits={"orders_per_sec":100}, retry_cfg={"max_attempts":1,"base_delay_ms":1}, dry_run=True, safety=safety, metrics=metrics, simulator_cfg={"reject_prob": 0.5, "partial_fill_prob":0.0})
        plan = {"ts":1, "symbol":"BTCUSDT", "side":"BUY", "price":100, "version":1, "_decision_monotonic": time.perf_counter()}
        await q.put(plan)
        task = asyncio.create_task(exec_.run())
        await asyncio.sleep(0.1)
        exec_.stop(); task.cancel()
        # order should be rejected
        oid = make_client_order_id(plan)
        rec = store.get_order(oid)
        assert rec and rec["status"] == "REJECTED"
        assert metrics.counters.get("orders_errors",0) >= 1
    finally:
        random.random = random_orig


def test_metrics_csv_export(tmp_path):
    m = Metrics()
    m.latency_tick_to_decision.observe(10)
    m.latency_decision_to_order.observe(5)
    csv_path = tmp_path/"metrics.csv"
    m.export_csv(str(csv_path))
    m.export_csv(str(csv_path))
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) >= 2 and rows[0][0] == "uptime_sec"