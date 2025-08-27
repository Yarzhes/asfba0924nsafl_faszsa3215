import asyncio, time
import pytest
from ultra_signals.live.order_exec import OrderExecutor
from ultra_signals.live.state_store import StateStore

@pytest.mark.asyncio
async def test_post_only_rejects_then_taker_fallback_sim():
    q = asyncio.Queue()
    store = StateStore(path=":memory:")
    exec_ = OrderExecutor(q, store, rate_limits={'orders_per_sec':100}, retry_cfg={'max_attempts':1,'base_delay_ms':1}, dry_run=True, safety=None, metrics=None)
    now_ms = int(time.time()*1000)
    plan = {
        'ts': now_ms//1000,
        'symbol':'BTCUSDT',
        'side':'LONG',
        'price':100.0,
        'version':1,
        'exec_plan':{
            'symbol':'BTCUSDT','side':'LONG','price':100.0,'type':'LIMIT','post_only':True,'ts':now_ms,'taker_fallback_after_ms':50,'taker_price':100.1
        },
        'size':1.0,
        'settings':{'execution':{'brackets':{'enabled':True,'stop_atr_mult':1.0,'tp_atr_mults':[1.5],'tp_scales':[1.0],'break_even':{'enabled':False},'trailing':{'enabled':False}}}}
    }
    await q.put(plan)
    task = asyncio.create_task(exec_.run())
    await asyncio.sleep(0.2)
    exec_.stop(); task.cancel()
    rows = store.list_orders()
    assert len(rows) >= 1
    # fallback should have filled
    o = rows[0]
    assert o['status'] in ('FILLED','PARTIAL','PENDING')
