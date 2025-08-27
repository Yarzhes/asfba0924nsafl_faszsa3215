from ultra_signals.events import classifier, gating, store
from ultra_signals.persist.db import init_db
from ultra_signals.persist.migrations import apply_migrations


def make_settings():
    return {
        'event_risk': {
            'enabled': True,
            'pre_window_minutes': {'HIGH': 90,'MED':45,'LOW':15},
            'post_window_minutes': {'HIGH': 90,'MED':30,'LOW':10},
            'actions': {
                'HIGH': {'mode':'VETO'},
                'MED': {'mode':'DAMPEN','size_mult':0.5},
                'LOW': {'mode':'DAMPEN','size_mult':0.75}
            },
            'missing_feed_policy': 'OPEN'
        }
    }


def test_classifier_mapping_basic():
    raw = {'id':'1','provider':'econ','name':'US CPI YoY','start_ts':1000,'end_ts':1000,'importance':3}
    c = classifier.classify(raw)
    assert c['category'] == 'CPI'
    assert c['importance'] == 3
    assert c['symbol_scope'] == 'GLOBAL'


def test_gate_veto_and_dampen(tmp_path):
    init_db(str(tmp_path / 'test.db'))
    apply_migrations()
    # Insert two events: one HIGH active now, one MED later
    now_ms = 1_000_000
    events = [
        {'id':'e1','provider':'econ','name':'US CPI YoY','category':'CPI','importance':3,'symbol_scope':'GLOBAL','start_ts':now_ms,'end_ts':now_ms,'source_payload':None},
        {'id':'e2','provider':'econ','name':'PMI print','category':'PMI','importance':2,'symbol_scope':'GLOBAL','start_ts':now_ms+200_000,'end_ts':now_ms+200_000,'source_payload':None},
    ]
    store.upsert_events(events)
    settings = make_settings()
    g1 = gating.evaluate('BTCUSDT', now_ms, None, None, settings)
    assert g1.action == 'VETO'
    g2 = gating.evaluate('BTCUSDT', now_ms+200_000, None, None, settings)
    # second event not inside pre window (MED pre=45min) but exact start -> should DAMPEN
    assert g2.action in ('DAMPEN','VETO')  # if overlapping logic picks first


def test_missing_feed_policy_safe():
    # Use fresh DB so previously inserted events do not leak into this test
    from ultra_signals.persist import db as _dbmod
    _dbmod._conn = None  # reset global connection for isolation
    gating.reset_caches()
    from ultra_signals.persist.db import init_db as _init
    import tempfile, os
    tmp = tempfile.mktemp(prefix='eventstest', suffix='.db')
    _init(tmp)
    settings = make_settings()
    settings['event_risk']['missing_feed_policy'] = 'SAFE'
    g = gating.evaluate('BTCUSDT', 0, None, None, settings)
    assert g.action in ('NONE','DAMPEN')  # SAFE may dampen without events
