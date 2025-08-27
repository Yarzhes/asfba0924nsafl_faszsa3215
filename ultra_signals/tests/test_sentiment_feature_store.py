from ultra_signals.sentiment import SentimentEngine
from ultra_signals.core.feature_store import FeatureStore

BASIC_SETTINGS = {
    'runtime': {'symbols': ['BTCUSDT']},
    'features': {'warmup_periods': 2},
    'sentiment': {
        'enabled': True,
        'symbols': ['BTCUSDT'],
        'symbol_keyword_map': {'BTCUSDT': ['btc']},
        'sources': {'twitter': {'enabled': False}, 'reddit': {'enabled': False}, 'funding': {'enabled': False}, 'fear_greed': {'enabled': False}},
        'telemetry': {'emit_metrics': False}
    }
}

def test_sentiment_snapshot_persisted():
    fs = FeatureStore(warmup_periods=5, settings=BASIC_SETTINGS)
    eng = SentimentEngine(BASIC_SETTINGS, feature_store=fs)
    # simulate ingest
    items = [{'ts': 1, 'text': 'btc pump', 'symbols': ['BTCUSDT'], 'meta': {}}]
    for it in items:
        it['polarity'] = 0.9
    eng.aggregator.ingest(items)
    eng.step()  # process pipeline (no network sources)
    snap = fs.get_sentiment_snapshot('BTCUSDT')
    assert snap is not None
    assert 'sent_score_s' in snap or 'sent_score_m' in snap
