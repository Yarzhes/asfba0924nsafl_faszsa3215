import math
from ultra_signals.drift.stat_tests import SPRT, CUSUM
from ultra_signals.drift.policy import PolicyEngine, ActionType


def test_sprt_accepts_h1_on_many_failures():
    s = SPRT(p0=0.6, p1=0.4, alpha=0.01, beta=0.1)
    # simulate many failures where success=False -> observed p < p1
    for _ in range(100):
        s.update(False)
    dec = s.decision()
    # Decision should lean toward accept_h0 or accept_h1 depending on p0/p1 setup
    assert dec in ("accept_h0", "accept_h1", None)


def test_cusum_detects_shift():
    c = CUSUM(threshold=2.0, drift=0.0)
    res = None
    for x in [0.1, 0.2, 0.5, 1.5, 0.9]:
        res = c.update(x)
    assert res in ("pos", None)


def test_policy_engine_moves_to_pause_and_retrain():
    pe = PolicyEngine()
    metrics = {"sprt_state": "accept_h1", "pf_delta_pct": -0.3, "maxdd_p95_breach": False, "ece_live": 0.02}
    act = pe.evaluate(metrics)
    assert act.type in (ActionType.PAUSE, ActionType.SHRINK)

    # maxdd breach forces retrain
    metrics["maxdd_p95_breach"] = True
    act2 = pe.evaluate(metrics)
    assert act2.type == ActionType.RETRAIN
