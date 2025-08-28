import os
import tempfile
import pandas as pd
from ultra_signals.backtest.event_runner import EventRunner, MockSignalEngine
from ultra_signals.backtest.data_adapter import DataAdapter
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.data.funding_provider import FundingProvider
from ultra_signals.features.derivatives_posture import compute_derivatives_posture


def _make_ohlcv_csv(path):
    # simple 5-bar series at 1m
    idx = pd.date_range('2025-01-01', periods=5, freq='1min')
    df = pd.DataFrame({
        'open': [100, 101, 102, 101, 103],
        'high': [101, 102, 103, 103, 104],
        'low':  [99, 100, 101, 100, 102],
        'close':[101, 102, 102, 103, 103],
        'volume':[10,11,12,13,14]
    }, index=idx)
    df.index.name = 'timestamp'
    df.to_parquet(path)


def test_backtest_funding_replay(tmp_path):
    # prepare temp data dir
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    ohlcv_file = data_dir / 'BTCUSDT_1m.parquet'
    _make_ohlcv_csv(ohlcv_file)

    # funding CSV
    funding_csv = tmp_path / 'funding_export.csv'
    df_f = pd.DataFrame([
        {'symbol':'BTCUSDT','ts': int(pd.Timestamp('2025-01-01T00:02:00').value//1_000_000), 'funding_rate': 0.0005, 'oi_notional': 1_000_000, 'venue':'test'},
        {'symbol':'BTCUSDT','ts': int(pd.Timestamp('2025-01-01T00:04:00').value//1_000_000), 'funding_rate': -0.0003, 'oi_notional': 900_000, 'venue':'test'},
    ])
    df_f.to_csv(funding_csv, index=False)

    # config for data adapter
    cfg = {'data': {'provider': 'parquet', 'base_path': str(data_dir)}}
    da = DataAdapter(cfg)

    # FeatureStore with FundingProvider wired
    fp = FundingProvider({'refresh_interval_minutes': 60})
    store = FeatureStore(warmup_periods=8, funding_provider=fp, settings={})

    # runner settings: point funding_replay path to our file
    settings = {'backtest': {'start_date': '2025-01-01', 'end_date': '2025-01-02', 'funding_replay': {'enabled': True, 'path': str(funding_csv)}, 'output_dir': str(tmp_path)}}

    engine = MockSignalEngine()
    runner = EventRunner(settings, da, engine, store)

    trades, equity = runner.run('BTCUSDT', '1m')

    # After run, funding provider should have loaded replay
    history = store.get_funding_rate_history('BTCUSDT')
    assert history is not None and len(history) >= 2

    # compute posture directly at end time
    last_ts = int(pd.Timestamp('2025-01-01T00:04:00').value//1_000_000)
    posture = compute_derivatives_posture(store, 'BTCUSDT', timestamp_ms=last_ts)
    # basic sanity checks
    assert hasattr(posture, 'funding_now')
    assert hasattr(posture, 'oi_notional')
