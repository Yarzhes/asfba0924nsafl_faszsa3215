import sys
import os


def run():
    # ensure project root on path for imports
    root = os.path.dirname(os.path.dirname(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    ok = True
    # import and run lvar tests
    try:
        from ultra_signals.tests.test_lvar import test_lvar_monotonic_with_shallower_depth, test_ttl_increases_with_lower_pr
        test_lvar_monotonic_with_shallower_depth()
        test_ttl_increases_with_lower_pr()
        print('lvar tests passed')
    except Exception as e:
        print('lvar tests failed:', e)
        ok = False

    try:
        from ultra_signals.tests.test_exec_adapter import test_veto_on_high_liq_cost, test_twap_on_high_lvar
        test_veto_on_high_liq_cost()
        test_twap_on_high_lvar()
        print('exec_adapter tests passed')
    except Exception as e:
        print('exec_adapter tests failed:', e)
        ok = False

    if not ok:
        sys.exit(2)

if __name__ == '__main__':
    run()
