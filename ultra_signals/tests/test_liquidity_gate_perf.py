import time
from ultra_signals.engine.gates.liquidity_gate import LiquidityGate, evaluate_gate
from ultra_signals.market.book_health import BookHealth

# Simple performance micro-benchmark; skips if environment extremely slow.

def test_liquidity_gate_perf_budget():
    settings = {
        'micro_liquidity': {
            'enabled': True,
            'profiles': {
                'trend': {
                    'spread_cap_bps': 15,
                    'impact_cap_bps': 40,
                    'rv_cap_bps': 25,
                    'rv_whip_cap_bps': 40,
                    'dr_skew_cap': 0.8,
                    'dampen': {'size_mult': 0.5}
                }
            },
            'cooldown_after_veto_secs': 0,
        }
    }
    gate = LiquidityGate(settings)
    bh = BookHealth(ts=int(time.time()), symbol='BTCUSDT', spread_bps=5.0, dr=0.1, impact_50k=10.0, rv_5s=8.0, mt=0.2)
    n = 2000
    start = time.perf_counter()
    now = int(time.time())
    for i in range(n):
        evaluate_gate('BTCUSDT', now + i, 'trend', bh, settings, gate)
    dur = time.perf_counter() - start
    avg_ms = (dur / n) * 1000.0
    # Allow generous margin for CI noise (<0.5ms target; assert <1.5ms)
    assert avg_ms < 1.5, f"Liquidity gate avg {avg_ms:.3f}ms exceeds budget"
