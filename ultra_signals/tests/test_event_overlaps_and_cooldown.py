from ultra_signals.events import store, gating
from ultra_signals.persist.db import init_db
from ultra_signals.persist.migrations import apply_migrations

SET = {
    'event_risk': {
        'enabled': True,
        'pre_window_minutes': {'HIGH': 10,'MED':5,'LOW':1},
        'post_window_minutes': {'HIGH':10,'MED':5,'LOW':1},
        'actions': {
            'HIGH': {'mode':'VETO'},
            'MED': {'mode':'DAMPEN','size_mult':0.5},
            'LOW': {'mode':'DAMPEN','size_mult':0.8}
        },
        'cooldown_minutes_after_veto': 5,
        'missing_feed_policy': 'OPEN'
    }
}

def setup_db(tmp_path):
    init_db(str(tmp_path / 'e.db'))
    apply_migrations()
    gating.reset_caches()

def test_overlapping_windows_high_wins(tmp_path):
    setup_db(tmp_path)
    now = 1_000_000
    # MED event window overlapping HIGH
    store.upsert_events([
        {'id':'h1','provider':'econ','name':'CPI','category':'CPI','importance':3,'symbol_scope':'GLOBAL','start_ts':now+60_000,'end_ts':now+60_000,'source_payload':None},
        {'id':'m1','provider':'econ','name':'PMI','category':'PMI','importance':2,'symbol_scope':'GLOBAL','start_ts':now+60_000,'end_ts':now+60_000,'source_payload':None},
    ])
    g = gating.evaluate('BTCUSDT', now+61_000, None, None, SET)
    assert g.action == 'VETO' and g.category == 'CPI'

def test_cooldown_applies(tmp_path):
    setup_db(tmp_path)
    base = 2_000_000
    store.upsert_events([
        {'id':'h2','provider':'econ','name':'CPI','category':'CPI','importance':3,'symbol_scope':'GLOBAL','start_ts':base,'end_ts':base,'source_payload':None}
    ])
    # evaluate 5 minutes before start inside pre-window (10m)
    g1 = gating.evaluate('ETHUSDT', base - 5*60*1000, None, None, SET)
    assert g1.action == 'VETO'
    # within 2 minutes still cooldown
    g2 = gating.evaluate('ETHUSDT', base + 2*60*1000, None, None, SET)
    assert g2.action == 'VETO' and g2.reason in ('COOLDOWN','CPI:HIGH')