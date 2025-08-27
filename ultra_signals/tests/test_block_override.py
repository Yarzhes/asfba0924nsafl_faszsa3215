from ultra_signals.collectors.cex_blocks import CEXBlockCollector
from ultra_signals.core.feature_store import FeatureStore

def test_block_override_detection():
    store = FeatureStore(warmup_periods=2, settings={'features': {'warmup_periods':2, 'whales': {'enabled': True}}})
    cfg = {
        'min_block_notional_usd': 1_000_000_000,  # absurdly high default so only override works
        'block_notional_overrides': {'BTCUSDT': 1000.0}
    }
    coll = CEXBlockCollector(store, ['BTCUSDT'], cfg)
    # feed 30 small trades so p99 is small
    for _ in range(30):
        coll.on_trade('BTCUSDT', price=10, qty=10, side='BUY')  # notional 100
    # large trade above override threshold
    coll.on_trade('BTCUSDT', price=100, qty=100, side='BUY')  # notional 10_000
    # verify a block record stored
    blocks = store._whale_state.get('blocks', {}).get('records', [])  # direct access for test
    assert any(r.get('notional') == 100*100 for r in blocks), f"Expected block trade in records: {blocks}"
