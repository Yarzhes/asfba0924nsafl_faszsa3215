"""Backtest parity golden day test (Sprint 39). Placeholder asserts pipeline stability.

This test re-loads a short deterministic OHLCV window twice and ensures no divergence
in validation stats (duplicates/gaps handling idempotent).
"""
import pandas as pd
from ultra_signals.backtest.data_adapter import DataAdapter
from ultra_signals.dq import validators

def _mock_settings():
    return {'data_quality': {'enabled': True, 'gap_policy': {'min_bar_coverage_pct': 80}}}


def test_backtest_parity_golden_day(tmp_path):
    # synthetic deterministic dataset
    rows = []
    ts = 0
    for i in range(10):
        rows.append({'timestamp': pd.to_datetime('2025-01-01') + pd.Timedelta(minutes=i), 'open':1,'high':1,'low':1,'close':1,'volume':1})
    df = pd.DataFrame(rows).set_index('timestamp')
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    df.to_csv(data_dir / 'BTCUSDT_1m.csv')
    adapter = DataAdapter({'data': {'provider':'csv','base_path': str(data_dir)}})
    run1 = adapter.load_ohlcv('BTCUSDT','1m','2025-01-01','2025-01-02')
    run2 = adapter.load_ohlcv('BTCUSDT','1m','2025-01-01','2025-01-02')
    assert run1.equals(run2)
    # Basic validation parity
    w1 = run1.reset_index().rename(columns={'timestamp':'ts'})
    w1['ts'] = pd.to_datetime(w1['ts']).astype('int64')//1_000_000
    rep1 = validators.validate_ohlcv_df(w1[['ts','open','high','low','close','volume']], 60_000, _mock_settings(), 'BTCUSDT', 'BACKTEST')
    w2 = run2.reset_index().rename(columns={'timestamp':'ts'})
    w2['ts'] = pd.to_datetime(w2['ts']).astype('int64')//1_000_000
    rep2 = validators.validate_ohlcv_df(w2[['ts','open','high','low','close','volume']], 60_000, _mock_settings(), 'BTCUSDT', 'BACKTEST')
    assert rep1.stats.get('rows') == rep2.stats.get('rows')
