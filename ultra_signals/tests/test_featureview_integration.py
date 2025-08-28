import os
import tempfile
from ultra_signals.orderflow.persistence import FeatureViewWriter


def test_featureview_write_and_update():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        fw = FeatureViewWriter(sqlite_path=path)
        slice_id = 'test-slice-123'
        rec = {
            'ts': 123456789,
            'symbol': 'X',
            'components': {'exec_strategy': 'VWAP'},
            'expected_cost_bps': 1.23,
            'realized_cost_bps': None,
            'price': 100.0,
            'slice_id': slice_id,
        }
        fw.write_record(rec)
        rows = fw.query_recent(limit=5)
        assert any(r.get('slice_id') == slice_id for r in rows)
        # update realized
        fw.update_by_slice_id(slice_id, {'realized_cost_bps': 5.5})
        rows2 = fw.query_recent(limit=5)
        updated = next((r for r in rows2 if r.get('slice_id') == slice_id), None)
        assert updated is not None
        assert float(updated.get('realized_cost_bps')) == 5.5
        fw.close()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
