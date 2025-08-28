import time
from ultra_signals.drift.policy import PolicyEngine, ActionType


def test_thresholds_and_shrink():
    pe = PolicyEngine()
    metrics = {'sprt_state': 'accept_h1', 'pf_delta_pct': -0.1, 'ece_live': 0.0, 'slip_delta_bps': 0.0, 'symbol': 'X'}
    a = pe.evaluate(metrics)
    assert a.type == ActionType.SHRINK
    assert 0.0 < a.size_mult < 1.0


def test_pause_hysteresis():
    cfg = {'hysteresis': {'pause_min_seconds': 60}}
    pe = PolicyEngine(cfg=cfg)
    # trigger pause by multiple codes
    metrics1 = {'sprt_state': 'accept_h1', 'pf_delta_pct': -0.3, 'symbol': 'S1', 'now': time.time()}
    a1 = pe.evaluate(metrics1)
    assert a1.type in (ActionType.PAUSE, ActionType.RETRAIN)

    # immediate re-evaluate should keep paused due to hysteresis
    metrics2 = {'sprt_state': None, 'pf_delta_pct': -0.1, 'symbol': 'S1', 'now': time.time()}
    a2 = pe.evaluate(metrics2)
    assert a2.type == ActionType.PAUSE

