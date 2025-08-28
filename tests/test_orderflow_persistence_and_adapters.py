import os
import time
from ultra_signals.orderflow.persistence import FeatureViewWriter
from ultra_signals.orderflow.engine import OrderflowEngine
from ultra_signals.orderflow.adapters.binance import BinanceAdapter


def test_feature_writer_sqlite(tmp_path):
    p = tmp_path / "ofs.db"
    w = FeatureViewWriter(str(p))
    eng = OrderflowEngine({"cvd_window": 60})
    now = int(time.time())
    # ingest a trade and compute score
    eng.ingest_trade(now, 100.0, 10, 'buy', aggressor=True)
    rec = {'ts': now, 'symbol': 'TEST', 'of_micro_score': 0.5, 'components': {'cvd': 0.5}, 'cvd': 10.0, 'imbalance': 0.1, 'tape_vps': 1.0, 'footprint': {100.0: 10}}
    w.write_record(rec)
    rows = w.query_recent(5)
    assert rows and rows[0]['symbol'] == 'TEST'
    w.close()


def test_binance_adapter_feeds_engine():
    eng = OrderflowEngine({"cvd_window": 60})
    ad = BinanceAdapter(eng, symbols=['TEST'])
    ad.start(count=5, interval=0)
    # after feeding, engine should have non-zero cvd
    s = eng.get_cvd()
    assert 'cvd_abs' in s
