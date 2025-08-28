"""Isolated smoke test: import TelemetryLogger, persist helpers, and PolicyEngine.

Run directly to validate emission, DB write (to temp DB), and retrain queue write.
"""
import tempfile, os, time, json, sys

# Ensure repo root is on sys.path so 'ultra_signals' package is importable
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ultra_signals.routing.telemetry import TelemetryLogger
from ultra_signals.drift.policy import PolicyEngine
from ultra_signals.persist import db


def run():
    # init db to temp file
    tmpdb = os.path.join(tempfile.gettempdir(), f"test_policy_smoke_{int(time.time())}.db")
    db.init_db(tmpdb)
    # create minimal policy_actions table so tests can insert
    try:
        db.execute(
            """CREATE TABLE IF NOT EXISTS policy_actions(
                   ts INTEGER, symbol TEXT, action_type TEXT, size_mult REAL, reason_codes TEXT, meta TEXT
               )"""
        )
    except Exception:
        pass

    tel = TelemetryLogger()
    pe = PolicyEngine()

    metrics = {'sprt_state': 'accept_h1', 'pf_delta_pct': -0.3, 'ece_live': 0.0, 'slip_delta_bps': 0.0, 'symbol': 'TEST', 'now': time.time()}
    action = pe.evaluate(metrics)

    # emit telemetry (use dict representation)
    act_dict = {'type': action.type.value, 'size_mult': action.size_mult, 'reason_codes': action.reason_codes, 'timestamp': action.timestamp}
    tel.emit_policy_action('TEST', 0, act_dict, metrics)

    # persist action
    db.record_policy_action('TEST', int(action.timestamp * 1000), action.type.value, action.size_mult, action.reason_codes, meta=metrics)

    # if retrain asked, write retrain job
    if action.type.value == 'retrain':
        qdir = os.path.join(tempfile.gettempdir(), 'retrain_queue')
        job = {'symbol': 'TEST', 'reason': action.reason_codes, 'ts': int(time.time() * 1000)}
        db.write_retrain_job(qdir, job)

    print('Telemetry events:', [e.decision for e in tel.get_events()])
    rows = db.fetchall('SELECT * FROM policy_actions')
    print('DB policy_actions rows:', rows)


if __name__ == '__main__':
    run()
