from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.events import KlineEvent
from ultra_signals.patterns.overlay import ascii_overlay


def make_bar(ts_ms, o, h, l, c, v=100):
    return KlineEvent(event_type="kline", timestamp=ts_ms, symbol="BTCUSDT", timeframe="5m", open=o, high=h, low=l, close=c, volume=v, closed=True)


def test_classical_double_top_and_overlay():
    settings = {
        'features': {'warmup_periods': 40},
        'patterns': {'enabled': True, 'classical': {'min_len': 30}}
    }
    store = FeatureStore(warmup_periods=50, settings=settings)
    base = 1_700_000_000_000
    # fabricate a double top: ramp, first peak, pullback, second peak same zone
    prices = [100 + i*0.2 for i in range(10)] + [102.5,102.7,102.6] + [101.5,101.2,101.4] + [102.5,102.6,102.55] + [101.8,101.6,101.4,101.2,101.0]
    for i,p in enumerate(prices):
        ts = base + i*300_000
        store.ingest_event(make_bar(ts, p-0.3, p, p-0.5, p-0.1, 500))
    feats = store.get_latest_features('BTCUSDT','5m')
    pats = feats.get('patterns') if feats else None
    assert pats, 'expected at least one pattern'
    # ensure a double top or head & shoulders candidate present
    kinds = {p.pat_type.value for p in pats}
    assert 'double_top' in kinds or 'head_shoulders' in kinds
    ohlcv = store.get_ohlcv('BTCUSDT','5m')
    art = ascii_overlay(ohlcv, pats)
    assert 'neckline' in art or 'breakout' in art