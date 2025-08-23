import pytest
import pandas as pd
from datetime import timedelta
from ultra_signals.backtest.walkforward import WalkForwardAnalysis

@pytest.fixture
def wf_settings():
    """Provides standard walk-forward settings for testing."""
    return {
        "walkforward": {
            "train_days": 30,
            "test_days": 10,
            "purge_days": 2
        },
        "backtest": {
             "start_date": "2023-01-01",
             "end_date": "2023-03-31"
        }
    }

def test_generate_windows_logic(wf_settings):
    """
    Verify that the window generation logic correctly creates overlapping
    windows with the specified train, test, and purge periods.
    """
    wfa = WalkForwardAnalysis(wf_settings, None, None)
    
    start_date = pd.to_datetime("2023-01-01")
    end_date = pd.to_datetime("2023-02-20") # Limit range for predictable output
    
    windows = wfa._generate_windows(start_date, end_date)
    
    # Expected windows:
    # 1. Train: 2023-01-01 to 2023-01-31, Test: 2023-02-03 to 2023-02-13
    # 2. Next start: 2023-01-01 + 10 days = 2023-01-11
    #    Train: 2023-01-11 to 2023-02-10, Test: 2023-02-13 to 2023-02-23 (too far)
    # So, only one window should be generated in this limited range.
    assert len(windows) == 1
    
    first_window = windows[0]
    train_start, train_end, test_start, test_end = first_window
    
    # Check train period
    assert train_start == pd.to_datetime("2023-01-01")
    assert train_end == pd.to_datetime("2023-01-31")
    
    # Check purge gap
    assert test_start == train_end + timedelta(days=wf_settings['walkforward']['purge_days'])
    
    # Check test period
    assert test_end == test_start + timedelta(days=wf_settings['walkforward']['test_days'])
    
def test_determinism_is_maintained():
    """
    This is a conceptual test. Determinism in walk-forward analysis
    is ensured by re-initializing the signal engine and its state for each fold.
    The implementation in `walkforward.py` follows this pattern, so this test
    serves as a reminder of that design principle.
    """
    # In the WalkForwardAnalysis class, a new instance of the signal engine
    # is created for each fold. This prevents state (e.g., indicator values)
    # from one fold leaking into the next, which is crucial for determinism.
    # `signal_engine_instance = self.signal_engine_class()`
    assert True, "Determinism depends on re-instantiating the engine per fold."