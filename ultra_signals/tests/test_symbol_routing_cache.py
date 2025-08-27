import pandas as pd
from pathlib import Path
from ultra_signals.backtest.data_adapter import DataAdapter


def test_symbol_routing_identity(tmp_path):
    # Prepare minimal csv file
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-10-01 00:00:00','2023-10-01 00:05:00']),
        'open':[1,2], 'high':[2,3], 'low':[0.5,1.5], 'close':[1.5,2.5], 'volume':[10,20]
    })
    df.to_csv(data_dir / 'BTCUSDT_5m.csv', index=False)
    cfg = {
        'data': {'provider':'csv', 'base_path': str(data_dir)},
        'runtime': {'symbol_routing': {'BTCUSDT':'BTCUSDT'}},
        'backtest': {'data': {'cache_path': str(tmp_path / '.cache')}} ,
        'batch_run': {'reuse_cache': True}
    }
    adapter = DataAdapter(cfg)
    out1 = adapter.load_ohlcv('BTCUSDT','5m','2023-10-01','2023-10-02')
    assert out1 is not None and len(out1)==2
    # Second load should hit cache even if we delete original file
    (data_dir / 'BTCUSDT_5m.csv').unlink()
    out2 = adapter.load_ohlcv('BTCUSDT','5m','2023-10-01','2023-10-02')
    assert out2 is not None and len(out2)==2


def test_cache_key_stable(tmp_path):
    data_dir = tmp_path / 'data'; data_dir.mkdir()
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-10-01 00:00:00','2023-10-01 00:05:00']),
        'open':[1,2], 'high':[2,3], 'low':[0.5,1.5], 'close':[1.5,2.5], 'volume':[10,20]
    })
    df.to_csv(data_dir / 'ETHUSDT_5m.csv', index=False)
    cfg = {'data': {'provider':'csv', 'base_path': str(data_dir)}, 'backtest': {'data': {'cache_path': str(tmp_path / '.cache')}}, 'batch_run': {'reuse_cache': True}}
    adapter = DataAdapter(cfg)
    a = adapter.load_ohlcv('ETHUSDT','5m','2023-10-01','2023-10-02')
    b = adapter.load_ohlcv('ETHUSDT','5m','2023-10-01','2023-10-02')
    assert a is not None and b is not None
    # Both loads should be equal
    assert a.equals(b)
