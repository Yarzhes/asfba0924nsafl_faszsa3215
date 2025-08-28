import asyncio
import time
from ultra_signals.live.engine_worker import EngineWorker
from ultra_signals.core.events import AggTradeEvent, KlineEvent


def make_feed_ref_with_vpin_policy(mode='veto', hi=0.0, lo=0.0, size_mult=0.5):
    class FR:
        settings = {
            'features': {
                'vpin': {
                    'policy': {'mode': mode, 'hi_th': hi, 'lo_th': lo, 'size_mult': size_mult},
                    'V_bucket': 10,
                    'K_buckets': 2,
                }
            }
        }

    return FR()


def run_engine_with_events(ew, events, timeout_s=0.1):
    async def runner():
        for e in events:
            await ew.in_queue.put(e)

        async def stopper():
            await asyncio.sleep(timeout_s)
            # put a dummy event to unblock the engine loop if it's waiting on queue.get()
            await ew.in_queue.put(AggTradeEvent(symbol='__SENTINEL__', price=0.0, quantity=0.0, timestamp=int(time.time()), is_buyer_maker=False))
            ew.stop()

        asyncio.create_task(stopper())
        await ew.run()

    asyncio.get_event_loop().run_until_complete(runner())


def test_engine_vpin_veto_policy(monkeypatch):
    qin = asyncio.Queue()
    qout = asyncio.Queue()
    ew = EngineWorker(qin, qout)
    ew.feed_ref = make_feed_ref_with_vpin_policy(mode='veto', hi=0.0, lo=0.0)

    at = AggTradeEvent(symbol='BTCUSD', price=100.0, quantity=10.0, timestamp=int(time.time()), is_buyer_maker=False)
    k = KlineEvent(symbol='BTCUSD', timeframe='1m', closed=True, timestamp=int(time.time()), open=100, high=101, low=99, close=100, volume=10.0)

    run_engine_with_events(ew, [at, k], timeout_s=0.05)
    assert qout.empty()


def test_engine_vpin_resize_policy(monkeypatch):
    qin = asyncio.Queue()
    qout = asyncio.Queue()
    ew = EngineWorker(qin, qout)
    ew.feed_ref = make_feed_ref_with_vpin_policy(mode='resize', hi=0.0, lo=0.0, size_mult=0.3)

    at = AggTradeEvent(symbol='BTCUSD', price=100.0, quantity=10.0, timestamp=int(time.time()), is_buyer_maker=False)
    k = KlineEvent(symbol='BTCUSD', timeframe='1m', closed=True, timestamp=int(time.time()), open=100, high=101, low=99, close=100, volume=10.0)

    run_engine_with_events(ew, [at, k], timeout_s=0.05)
    assert not qout.empty()
    plan = qout.get_nowait()
    assert 'size_mult' in plan and plan['size_mult'] == 0.3
