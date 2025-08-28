import unittest
import sys
import os


def main():
    loader = unittest.TestLoader()
    tests_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir=tests_dir, pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
