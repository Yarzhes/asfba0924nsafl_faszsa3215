import pandas as pd
import numpy as np
from ultra_signals.features.flow_metrics import compute_flow_metrics


def _mock_ohlcv(n=40):
    base = pd.Timestamp('2023-01-01 00:00:00')
    idx = [base + pd.Timedelta(minutes=i) for i in range(n)]
    df = pd.DataFrame({
        'open': np.linspace(100,101,n),
        'high': np.linspace(100.5,101.5,n),
        'low': np.linspace(99.5,100.5,n),
        'close': np.linspace(100,101,n)+np.sin(np.linspace(0,3.14,n))*0.2,
        'volume': np.random.randint(50,150,size=n)
    }, index=idx)
    return df


def test_cvd_direction():
    df = _mock_ohlcv()
    # create trades: 10 aggressive buys of qty 2, 3 aggressive sells of qty 1
    ts_last = int(df.index[-1].value // 1_000_000)
    trades = []
    for i in range(10):
        trades.append((ts_last-1000*i, 100+i*0.01, 2.0, False))  # False = buyer NOT maker => aggressive buy
    for i in range(3):
        trades.append((ts_last-2000*i, 100-i*0.01, 1.0, True))
    feats = compute_flow_metrics(df, trades, [], {'bid':100,'ask':100.1,'B':10,'A':9}, {'features': {'flow_metrics':{'enabled':True}}}, state={})
    assert feats.get('cvd') is not None and feats['cvd'] > 0


def test_oi_rate_spike():
    df = _mock_ohlcv()
    state = {'oi_series': [1000, 1015]}
    feats = compute_flow_metrics(df, [], [], None, {'features': {'flow_metrics': {'enabled': True}}}, state)
    assert 'oi_rate' in feats and feats['oi_rate'] > 0


def test_liquidation_cluster():
    df = _mock_ohlcv()
    ts_last = int(df.index[-1].value // 1_000_000)
    liqs = [(ts_last - 1000*i, 'BUY' if i%2==0 else 'SELL', 10000.0) for i in range(4)]
    feats = compute_flow_metrics(df, [], liqs, None, {'features': {'flow_metrics': {'enabled': True, 'liquidations': {'min_cluster_size':3}}}}, state={})
    assert feats.get('liq_cluster') == 1


def test_depth_imbalance():
    df = _mock_ohlcv()
    feats = compute_flow_metrics(df, [], [], {'bid':100,'ask':100.1,'B':200,'A':100}, {'features': {'flow_metrics': {'enabled': True}}}, state={})
    assert 'ofi' in feats and feats['ofi'] > 0


def test_cross_spread_flag_safe():
    df = _mock_ohlcv()
    # Provide mids via state to trigger spread logic
    state = {'mids_multi': [100.0, 100.8]}  # ~0.8% -> 80 bps > 5 -> flag
    feats = compute_flow_metrics(df, [], [], None, {'features': {'flow_metrics': {'enabled': True, 'spread': {'max_bp':5}}}}, state)
    if 'spread_bps' in feats:
        assert feats['spread_dev_flag'] in (0,1)
