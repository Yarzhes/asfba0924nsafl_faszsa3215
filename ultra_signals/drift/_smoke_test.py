"""Quick smoke test for drift primitives.

Run with: python "c:/Users/Almir/Projects/Trading Helper/ultra_signals/drift/_smoke_test.py"
"""
import sys
sys.path.insert(0, r"c:\Users\Almir\Projects\Trading Helper")

from ultra_signals.drift.stat_tests import SPRT, CUSUM
from ultra_signals.drift.policy import PolicyEngine


def main():
    s = SPRT(p0=0.6, p1=0.4, alpha=0.01, beta=0.1)
    for _ in range(100):
        s.update(False)
    print('SPRT llr=', s.llr(), 'decision=', s.decision())

    c = CUSUM(threshold=2.0, drift=0.0)
    res = None
    for x in [0.1, 0.2, 0.5, 1.5, 0.9]:
        res = c.update(x)
    print('CUSUM result=', res)

    pe = PolicyEngine()
    metrics = {"sprt_state": "accept_h1", "pf_delta_pct": -0.3, "maxdd_p95_breach": False, "ece_live": 0.02}
    act = pe.evaluate(metrics)
    print('Policy action1=', act.type, 'size_mult=', act.size_mult, 'codes=', act.reason_codes)
    metrics['maxdd_p95_breach'] = True
    act2 = pe.evaluate(metrics)
    print('Policy action2=', act2.type, 'size_mult=', act2.size_mult, 'codes=', act2.reason_codes)


if __name__ == '__main__':
    main()
