import asyncio
import tempfile
import os
import time
from ultra_signals.live.engine_worker import EngineWorker
from ultra_signals.live.order_exec import OrderExecutor
from ultra_signals.orderflow.persistence import FeatureViewWriter


def test_pipeline_smoke(tmp_path):
    # temp DB for feature view
    fv_path = str(tmp_path / 'fv.db')
    fw = FeatureViewWriter(sqlite_path=fv_path)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    engine_in = asyncio.Queue()
    order_q = asyncio.Queue()
    # EngineWorker will consume plan from engine_in and push VWAP children to order_q
    engine = EngineWorker(engine_in, order_q, feature_writer=fw)
    # create a minimal fake feed_ref that exposes venue_router and snapshot
    from ultra_signals.routing.types import L2Book, PriceLevel, AggregatedBook
    class FakeVenueAdapter:
        def __init__(self, vid):
            self.vid = vid
            self.taker_fee = 1.0

    class FakeVenueRouter:
        def __init__(self):
            self.venues = {'A': FakeVenueAdapter('A')}

    class FakeFeed:
        def __init__(self):
            self.venue_router = FakeVenueRouter()
        def snapshot(self, symbol):
            # simple L2Book with depth
            bids = [PriceLevel(100.0, 1.0), PriceLevel(99.9, 1.0)]
            asks = [PriceLevel(100.1, 1.0), PriceLevel(100.2, 1.0)]
            book = L2Book(bids=bids, asks=asks, ts_ms=int(time.time()*1000))
            return AggregatedBook(symbol=symbol, books={'A': book})

    engine.feed_ref = FakeFeed()

    # Use OrderExecutor in dry-run mode with simulator to force fills
    store = None
    try:
        from ultra_signals.live.state_store import StateStore
        store = StateStore(':memory:')
    except Exception:
        store = None
    # order_sender None -> uses dry_run simulator
    executor = OrderExecutor(order_q, store, rate_limits={}, retry_cfg={}, dry_run=True, feature_writer=fw, simulator_cfg={'slippage_bps_min':0.0,'slippage_bps_max':0.0,'partial_fill_prob':0.0,'reject_prob':0.0})

    async def run_once():
        # start executor
        t_exec = asyncio.create_task(executor.run())
        # Directly create VWAPExecutor and generate slices
        from ultra_signals.routing.vwap_adapter import VWAPExecutor
        vexec = VWAPExecutor(venues={'A': None}, volume_curve=[1.0], pr_cap=1.0, feature_writer=fw)
        # attach fake feed as feature provider
        vexec.feature_provider = engine.feed_ref.snapshot if hasattr(engine.feed_ref, 'snapshot') else None
        slices = vexec.execute(engine.feed_ref, 'buy', 2.0, 'X')
        # create child plans like EngineWorker would and enqueue
        for sl in slices:
            child = {
                'ts': int(time.time()),
                'symbol': 'X',
                'side': 'LONG',
                'size': sl.get('slice_notional'),
                'price': sl.get('expected_price'),
                'parent_id': 123,
                'slice_id': sl.get('slice_id'),
                'exec_plan': {'order_type': 'MARKET' if sl.get('style') == 'MARKET' else 'LIMIT', 'expected_price': sl.get('expected_price')},
            }
            await order_q.put(child)
        # wait a short while for executor to process queue
        await asyncio.sleep(0.5)
        executor.stop()
        t_exec.cancel()
        try:
            await t_exec
        except Exception:
            pass

    loop.run_until_complete(run_once())
    rows = fw.query_recent(limit=20)
    # A parent plan record should be written by VWAPExecutor and at least one slice should have been written
    assert any(r.get('symbol') == 'X' for r in rows)
    # Also cleaned up
    fw.close()
    loop.close()
