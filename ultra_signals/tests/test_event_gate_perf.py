import time
from ultra_signals.events import evaluate_gate, reset_caches

def test_event_gate_perf_smoke():
    settings = {
        'event_risk': {
            'enabled': True,
            'pre_window_minutes': {},
            'post_window_minutes': {},
            'actions': {'HIGH': {'mode': 'VETO'}, 'MED': {'mode': 'DAMPEN','size_mult':0.5}, 'LOW': {'mode': 'NONE'}},
            'cooldown_minutes_after_veto': 0,
        }
    }
    reset_caches()
    # warm cache build
    now_ms = int(time.time()*1000)
    evaluate_gate('BTCUSDT', now_ms, None, None, settings)
    start = time.time()
    N = 2000
    for _ in range(N):
        evaluate_gate('BTCUSDT', now_ms, None, None, settings)
    dur = time.time()-start
    avg_ms = (dur / N) * 1000.0
    # loose upper bound (should be sub-ms on typical dev machine without events)
    assert avg_ms < 5.0, f"event gate evaluate too slow avg_ms={avg_ms:.3f}"  # pragma: no cover if fast