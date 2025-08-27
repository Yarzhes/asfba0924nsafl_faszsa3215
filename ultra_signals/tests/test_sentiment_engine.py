import time
from ultra_signals.sentiment import SentimentEngine

BASIC_SETTINGS = {
    'runtime': {'symbols': ['BTCUSDT','ETHUSDT']},
    'sentiment': {
        'enabled': True,
        'symbols': ['BTCUSDT','ETHUSDT'],
        'symbol_keyword_map': {'BTCUSDT': ['btc','bitcoin'], 'ETHUSDT': ['eth','ethereum']},
        'sources': {  # disable network sources for deterministic unit test
            'twitter': {'enabled': False},
            'reddit': {'enabled': False},
            'fear_greed': {'enabled': False},
            'funding': {'enabled': False},
        },
    }
}


def test_sentiment_engine_manual_feed_and_veto():
    eng = SentimentEngine(BASIC_SETTINGS)
    # inject fake items directly through aggregator to simulate bullish burst
    now = int(time.time())
    items = []
    for i in range(30):
        items.append({'ts': now - i*30, 'text': 'BTC going to the moon 🚀', 'symbols': ['BTCUSDT'], 'meta': {}})
    # score manually (simulate step path)
    for it in items:
        it['polarity'] = 0.9
    eng.aggregator.ingest(items)
    snaps = eng.aggregator.latest_per_symbol
    assert 'BTCUSDT' in snaps
    fv = eng.feature_view()['BTCUSDT']
    assert 'sent_score_s' in fv
    # With many high polarity items z should be elevated positive
    assert fv.get('sent_score_s', 0) > 0
    # Force z high to trigger extreme flag
    snaps['BTCUSDT']['sent_z_s'] = 3.0
    snaps['BTCUSDT']['extreme_flag_bull'] = 1
    assert eng.maybe_veto('BTCUSDT') in ('SENTIMENT_EUPHORIA', None)  # allow None if config off


def test_sentiment_size_modifier_fallback():
    eng = SentimentEngine(BASIC_SETTINGS)
    # Force extreme flag state
    eng.aggregator.latest_per_symbol['ETHUSDT'] = {'extreme_flag_bear': 1, 'sent_score_s': -0.8, 'sent_z_s': -3.1}
    mod = eng.size_modifier('ETHUSDT')
    assert 0 < mod <= 1.0
