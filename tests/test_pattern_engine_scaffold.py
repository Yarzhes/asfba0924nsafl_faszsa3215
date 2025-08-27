import pandas as pd
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.events import KlineEvent


def make_bar(ts_ms, o=100, h=101, l=99, c=100.5, v=10):
    return KlineEvent(event_type="kline", timestamp=ts_ms, symbol="BTCUSDT", timeframe="1m", open=o, high=h, low=l, close=c, volume=v, closed=True)


def test_pattern_engine_attaches_patterns():
    settings = {
        'features': {
            'warmup_periods': 5,
        },
        'patterns': {
            'enabled': True,
            'range_compression': {
                'window': 5,
                'max_band_pct': 0.06,
            }
        }
    }
    store = FeatureStore(warmup_periods=5, settings=settings)
    base_ts = 1_700_000_000_000
    # feed > window bars with small ranges to trigger compression detector
    for i in range(8):
        ts = base_ts + i * 60_000
        bar = make_bar(ts_ms=ts, o=100+i*0.01, h=100.5+i*0.01, l=99.5+i*0.01, c=100.1+i*0.01, v=50)
        store.ingest_event(bar)

    feats = store.get_latest_features("BTCUSDT", "1m")
    assert feats is not None
    patterns = feats.get('patterns')
    assert patterns is not None and len(patterns) >= 1
    # basic field sanity
    p0 = patterns[0]
    assert p0.pat_type.value in ('sym_triangle',)
    assert p0.stage.value in ('forming', 'confirmed', 'failed')
