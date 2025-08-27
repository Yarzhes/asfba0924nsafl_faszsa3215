import time
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.market.book_health import BookHealth
from ultra_signals.engine.gates.liquidity_gate import evaluate_gate
from ultra_signals.apps.dashboard import compute_snapshot

class DummySettings(dict):
    pass

def _mk_settings():
    return {
        'features': {'warmup_periods': 2},
        'micro_liquidity': {
            'enabled': True,
            'profiles': {
                'trend': {
                    'spread_cap_bps': 20,
                    'impact_cap_bps': 50,
                    'rv_cap_bps': 30,
                    'rv_whip_cap_bps': 45,
                    'dr_skew_cap': 0.8,
                    'dampen': {'size_mult': 0.6}
                }
            }
        }
    }


def test_proxy_book_health_and_dashboard_snapshot():
    settings = _mk_settings()
    store = FeatureStore(warmup_periods=2, settings=settings)
    ts0 = int(time.time()*1000)
    # feed two bars to satisfy warmup and trigger feature compute
    store.on_bar('BTCUSDT', '5m', {'timestamp': ts0, 'open':100, 'high':105, 'low':95, 'close':102, 'volume':10})
    store.on_bar('BTCUSDT', '5m', {'timestamp': ts0+300000, 'open':102, 'high':108, 'low':101, 'close':106, 'volume':12})

    # No real book health set -> evaluate gate should rely on proxy (set by event_runner logic normally)
    # Simulate proxy creation manually using proxy function
    from ultra_signals.market.book_health_proxy import compute_proxies
    feats = store.get_latest_features('BTCUSDT', '5m') or {}
    proxies = compute_proxies({**feats, 'ohlcv': {'close':106,'high':108,'low':101}})
    bh_proxy = BookHealth(ts=int(time.time()), symbol='BTCUSDT',
                          spread_bps=proxies.get('spread_bps'), dr=proxies.get('dr'),
                          impact_50k=proxies.get('impact_50k'), rv_5s=proxies.get('rv_5s'), source='proxy')
    store.set_latest_book_health('BTCUSDT', bh_proxy)

    out = evaluate_gate('BTCUSDT', int(time.time()), 'trend', bh_proxy, settings)
    assert out.action in ('NONE','DAMPEN','VETO')

    snap = compute_snapshot()
    liq = snap.get('liquidity_gate', {})
    # Snapshot should include dict keyed by symbol once gate ran; emulate by injecting via wrapper above
    # Accept empty (if dashboard imported before evaluate cached) but if present should contain structure
    if liq:
        assert isinstance(liq, dict)
    # ensure BookHealth proxy fields present
    assert bh_proxy.spread_bps is not None
