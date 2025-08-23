Title: [Tests] Low test coverage in real_engine.py and regime.py
Steps:
1. Run `pytest --cov=ultra_signals --cov-report=term-missing -q`
Expected:
Coverage for `ultra_signals/engine/real_engine.py` and `ultra_signals/engine/regime.py` should be at least 80%.
Actual:
Coverage for `ultra_signals/engine/real_engine.py` is 26%.
Coverage for `ultra_signals/engine/regime.py` is 25%.
Logs/Screens: (paths)
See pytest output.
Scope: both
Suspected cause: Missing unit tests for these modules.
Severity: med
Fix idea: Add comprehensive unit tests for `real_engine.py` and `regime.py` to increase code coverage.
Owner: Code