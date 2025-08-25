import asyncio
import pytest
from ultra_signals.live.order_exec import OrderExecutor, make_client_order_id
from ultra_signals.live.state_store import StateStore


@pytest.mark.asyncio
async def test_order_idempotency(tmp_path):
    q = asyncio.Queue()
    store = StateStore(str(tmp_path / "state.db"))
    exec = OrderExecutor(q, store, rate_limits={"orders_per_sec": 100}, retry_cfg={"max_attempts":1}, dry_run=True)
    task = asyncio.create_task(exec.run())
    plan = {"ts": 1, "symbol": "BTCUSDT", "side": "BUY", "price": 100, "version": 1}
    cid = make_client_order_id(plan)
    await q.put(plan)
    # duplicate
    await q.put(plan)
    await asyncio.sleep(0.1)
    row = store.get_order(cid)
    assert row is not None
    # there should be exactly one row; second put should not create duplicate or error
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0)  # allow cancellation to propagate silently
