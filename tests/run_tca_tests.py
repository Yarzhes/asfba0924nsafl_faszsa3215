import sys

if __name__ == '__main__':
    sys.path.insert(0, '.')
    # Directly import and run the test function (no pytest available in this environment)
    try:
        from tests import test_tca_engine
        test_tca_engine.test_slip_and_fill_ratio()
        print('TESTS PASSED')
        sys.exit(0)
    except AssertionError as e:
        print('TEST FAILED:', e)
        sys.exit(1)
    except Exception as e:
        print('ERROR RUNNING TESTS:', e)
        sys.exit(1)
