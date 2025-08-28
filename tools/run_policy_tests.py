import importlib, os, sys

# Ensure repo root on sys.path for in-repo imports
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def run():
    mod = importlib.import_module('ultra_signals.drift.tests.test_policy_hysteresis')
    try:
        mod.test_thresholds_and_shrink()
        mod.test_pause_hysteresis()
        print('policy tests OK')
    except AssertionError as e:
        print('test assertion failed:', e)
    except Exception as e:
        print('test error:', e)

if __name__ == '__main__':
    run()
