import time
from ultra_signals.features.whales import compute_whale_features
from ultra_signals.core.feature_store import FeatureStore


def test_compute_whale_features_basic():
    now = int(time.time()*1000)
    state = {
        'exchange_flows': {'records': [
            {'ts': now-1000, 'symbol':'BTCUSDT', 'direction':'WITHDRAWAL', 'usd': 2_000_000},
            {'ts': now-2000, 'symbol':'BTCUSDT', 'direction':'DEPOSIT', 'usd': 1_000_000},
        ]},
        'blocks': {'records': [ {'ts': now-30_000, 'symbol':'BTCUSDT', 'notional': 3_000_000, 'side':'SELL', 'type':'BLOCK'} ]},
        'options': {'snapshot': {'call_put_volratio_z': 1.5}},
        'smart_money': {'records': [ {'ts': now-10_000, 'symbol':'BTCUSDT', 'side':'BUY','usd':800_000} ], 'hit_rate_30d':0.6},
    }
    cfg = {'windows': {'short_sec':3600,'medium_sec':6*3600,'long_sec':24*3600}}
    out = compute_whale_features('BTCUSDT', now, state, cfg)
    assert 'whale_net_inflow_usd_s' in out and out['whale_net_inflow_usd_s'] is not None
    assert out.get('smart_money_hit_rate_30d') == 0.6


def test_deposit_withdrawal_burst_flags():
    now = int(time.time()*1000)
    # Create many small deposits baseline then one large recent deposit
    records = []
    for i in range(30):
        records.append({'ts': now-10_000- i*60_000, 'symbol':'BTCUSDT','direction':'DEPOSIT','usd':100_000})
    # Big recent deposit triggers burst
    records.append({'ts': now-30_000,'symbol':'BTCUSDT','direction':'DEPOSIT','usd':2_000_000})
    state = {'exchange_flows': {'records': records}}
    cfg = {'windows': {'short_sec':3600}, 'deposit_burst_multiplier':5.0}
    out = compute_whale_features('BTCUSDT', now, state, cfg)
    assert out.get('exch_deposit_burst_flag') == 1


def test_feature_store_whale_helpers():
    # Use minimal warmup so first bar triggers feature computation
    store = FeatureStore(warmup_periods=2, settings={'features': {'warmup_periods':2, 'whales': {'enabled': True}}})
    ts_ms = int(time.time()*1000)
    store.whale_add_exchange_flow('BTCUSDT','DEPOSIT',1_000_000,ts_ms)
    store.whale_add_exchange_flow('BTCUSDT','WITHDRAWAL',2_500_000,ts_ms)
    store.whale_add_block_trade('BTCUSDT','BUY',3_000_000,'BLOCK',ts_ms)
    store.whale_update_options_snapshot({'call_put_volratio_z':0.5})
    store.whale_add_smart_money_trade('BTCUSDT','BUY',800_000,'walletA',ts_ms)
    store.whale_set_smart_money_hit_rate(0.55)
    # Trigger a dummy bar to compute whales (enabled via settings override)
    # Ingest two bars to satisfy warmup
    bar_ts = ts_ms - 60_000
    store.on_bar('BTCUSDT','1m',[bar_ts,1,1,1,1,10])
    store.on_bar('BTCUSDT','1m',[ts_ms,1,1,1,1,12])
    feats = store.get_latest_features('BTCUSDT','1m')
    assert feats and 'whales' in feats
    wf = feats['whales']
    assert wf.smart_money_hit_rate_30d == 0.55
