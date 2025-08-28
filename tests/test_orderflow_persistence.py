import os
import tempfile

from ultra_signals.orderflow.persistence import FeatureViewWriter


def test_write_and_query_recent():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        w = FeatureViewWriter(sqlite_path=path)
        rec = {
            "ts": 1234567890,
            "symbol": "BTCUSDT",
            "of_micro_score": 0.42,
            "price": 42000.5,
            "cvd_abs": 100000.0,
            "cvd_pct": 0.12,
            "cvd_z": 2.5,
            "cvd_div_flag": 1,
            "ob_imbalance_top1": 0.05,
            "tape_vps": 5000.0,
            "tape_burst_flag": 0,
            "footprint_sr_level_px": 41950.0,
            "components": {"cvd_window": 60},
        }
        w.write_record(rec)
        out = w.query_recent(1)
        assert len(out) == 1
        r = out[0]
        assert r["symbol"] == "BTCUSDT"
        assert float(r["of_micro_score"]) == 0.42
        assert float(r["cvd_abs"]) == 100000.0
        assert r["components"]["cvd_window"] == 60
    finally:
        try:
            w.close()
        except Exception:
            pass
        os.remove(path)
