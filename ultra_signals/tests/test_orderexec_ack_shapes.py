import tempfile
import os
import asyncio
from ultra_signals.live.order_exec import OrderExecutor
from ultra_signals.live.state_store import StateStore
from ultra_signals.orderflow.persistence import FeatureViewWriter


def make_plan():
    return {
        'ts': 1,
        'symbol': 'X',
        'side': 'LONG',
        'size': 1.0,
        'price': 100.0,
        'slice_id': 'slice-ack-1',
        'exec_plan': {'expected_price': 100.0},
    }


def run_proc(loop, executor, plan):
    return loop.run_until_complete(executor._process_plan(plan))


def test_ack_shapes_dict_with_ack():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        fw = FeatureViewWriter(sqlite_path=path)
        store = StateStore(':memory:')
        # order_sender returns dict with 'ack' object-like
        def sender(plan, cid):
            class Ack:
                status = 'FILLED'
                avg_px = 101.0
                venue_order_id = 'V1'

            return {'ack': Ack(), 'venue': 'A'}

        # pre-write a FeatureView record for the slice so update_by_slice_id can find it
        fw.write_record({'ts': 1, 'symbol': 'X', 'components': {}, 'slice_id': 'slice-ack-1', 'expected_cost_bps': 0.0})
        # wrap update to detect invocation
        called = {'v': False}
        orig = fw.update_by_slice_id
        def wrap(sid, updates):
            called['v'] = True
            return orig(sid, updates)
        fw.update_by_slice_id = wrap
        ex = OrderExecutor(queue=asyncio.Queue(), store=store, rate_limits={}, retry_cfg={}, dry_run=False, order_sender=sender, feature_writer=fw)
        loop = asyncio.new_event_loop()
        try:
            run_proc(loop, ex, make_plan())
        finally:
            loop.close()

        rows = fw.query_recent(limit=5)
        assert any(r.get('slice_id') == 'slice-ack-1' for r in rows)
        # update should have written realized_cost_bps
        assert called['v'] is True
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def test_ack_shapes_flat_dict():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        fw = FeatureViewWriter(sqlite_path=path)
        store = StateStore(':memory:')

        def sender(plan, cid):
            return {'status': 'FILLED', 'avg_px': 102.0, 'venue': 'B'}

        fw.write_record({'ts': 1, 'symbol': 'X', 'components': {}, 'slice_id': 'slice-ack-1', 'expected_cost_bps': 0.0})
        called = {'v': False}
        orig = fw.update_by_slice_id
        def wrap(sid, updates):
            called['v'] = True
            return orig(sid, updates)
        fw.update_by_slice_id = wrap
        ex = OrderExecutor(queue=asyncio.Queue(), store=store, rate_limits={}, retry_cfg={}, dry_run=False, order_sender=sender, feature_writer=fw)
        loop = asyncio.new_event_loop()
        try:
            run_proc(loop, ex, make_plan())
        finally:
            loop.close()

        rows = fw.query_recent(limit=5)
        assert any(r.get('slice_id') == 'slice-ack-1' for r in rows)
        assert called['v'] is True
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def test_ack_shapes_object_ack():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        fw = FeatureViewWriter(sqlite_path=path)
        store = StateStore(':memory:')

        class AckObj:
            def __init__(self):
                self.status = 'FILLED'
                self.avg_px = 103.0
                self.venue_order_id = 'X'

        def sender(plan, cid):
            return AckObj()

        fw.write_record({'ts': 1, 'symbol': 'X', 'components': {}, 'slice_id': 'slice-ack-1', 'expected_cost_bps': 0.0})
        called = {'v': False}
        orig = fw.update_by_slice_id
        def wrap(sid, updates):
            called['v'] = True
            return orig(sid, updates)
        fw.update_by_slice_id = wrap
        ex = OrderExecutor(queue=asyncio.Queue(), store=store, rate_limits={}, retry_cfg={}, dry_run=False, order_sender=sender, feature_writer=fw)
        loop = asyncio.new_event_loop()
        try:
            run_proc(loop, ex, make_plan())
        finally:
            loop.close()

        rows = fw.query_recent(limit=5)
        assert any(r.get('slice_id') == 'slice-ack-1' for r in rows)
        assert called['v'] is True
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
