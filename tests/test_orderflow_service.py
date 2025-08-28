import os
import tempfile
import time

from ultra_signals.orderflow.service import SimulatedFeed, OrderflowService
from ultra_signals.orderflow.engine import OrderflowCalculator
from ultra_signals.orderflow.persistence import FeatureViewWriter


def test_service_run_once_writes_record():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        writer = FeatureViewWriter(sqlite_path=path)
        calc = OrderflowCalculator(vps_window_s=5, cvd_window_s=10)
        feed = SimulatedFeed(symbol="TST", base_price=50.0, seed=123)
        svc = OrderflowService(calc, writer, feed, interval_s=1.0, symbol="TST")
        rec = svc.run_once(ts=time.time())
        # query recent directly from writer
        out = writer.query_recent(1)
        assert len(out) == 1
        r = out[0]
        assert r["symbol"] == "TST"
        assert "components" in r
    finally:
        try:
            writer.close()
        except Exception:
            pass
        os.remove(path)
