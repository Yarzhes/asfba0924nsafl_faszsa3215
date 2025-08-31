#!/usr/bin/env python3
"""
Test script to verify that momentum features return valid values instead of NaN.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_signals.features.momentum import compute_momentum_features

def test_small_dataset():
    """Test momentum features with a small dataset (insufficient for traditional calculations)."""
    print("Testing momentum features with small dataset...")
    
    # Create a small OHLCV dataset (only 10 periods)
    dates = pd.date_range('2025-01-01', periods=10, freq='5min')
    ohlcv = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 101.5, 103.0, 102.0, 104.0, 103.5, 105.0, 104.0],
        'high': [101.0, 102.0, 103.0, 102.5, 104.0, 103.0, 105.0, 104.5, 106.0, 105.0],
        'low': [99.5, 100.5, 101.5, 101.0, 102.5, 101.5, 103.5, 103.0, 104.5, 103.5],
        'close': [100.5, 101.5, 102.5, 101.8, 103.2, 102.3, 104.1, 103.8, 105.2, 104.2],
        'volume': [1000, 1100, 1200, 1050, 1300, 1150, 1400, 1250, 1500, 1350]
    }, index=dates)
    
    features = compute_momentum_features(ohlcv)
    
    print(f"Small dataset features: {features}")
    
    # Check that no values are NaN
    for key, value in features.items():
        assert not np.isnan(value), f"{key} should not be NaN, got {value}"
        assert isinstance(value, (int, float)), f"{key} should be numeric, got {type(value)}"
    
    print("✓ Small dataset test passed - no NaN values")

def test_larger_dataset():
    """Test momentum features with a larger dataset (sufficient for calculations)."""
    print("\nTesting momentum features with larger dataset...")
    
    # Create a larger OHLCV dataset (50 periods)
    dates = pd.date_range('2025-01-01', periods=50, freq='5min')
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price data with some trend
    base_price = 100.0
    price_changes = np.random.normal(0, 0.5, 50).cumsum()
    closes = base_price + price_changes
    
    ohlcv = pd.DataFrame({
        'open': closes + np.random.normal(0, 0.1, 50),
        'high': closes + np.abs(np.random.normal(0, 0.3, 50)),
        'low': closes - np.abs(np.random.normal(0, 0.3, 50)),
        'close': closes,
        'volume': np.random.randint(1000, 2000, 50)
    }, index=dates)
    
    features = compute_momentum_features(ohlcv)
    
    print(f"Large dataset features: {features}")
    
    # Check that no values are NaN
    for key, value in features.items():
        assert not np.isnan(value), f"{key} should not be NaN, got {value}"
        assert isinstance(value, (int, float)), f"{key} should be numeric, got {type(value)}"
    
    # With sufficient data, RSI should be between 0 and 100
    assert 0 <= features['rsi'] <= 100, f"RSI should be between 0-100, got {features['rsi']}"
    
    print("✓ Large dataset test passed - valid momentum indicators")

if __name__ == "__main__":
    try:
        test_small_dataset()
        test_larger_dataset()
        print("\n🎉 All momentum feature tests passed! The fixes are working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
