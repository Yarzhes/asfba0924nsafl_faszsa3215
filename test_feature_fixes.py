#!/usr/bin/env python3
"""
Test script to verify that the feature calculation fixes resolve the NaN issue.
"""

import pandas as pd
import numpy as np
from ultra_signals.features.trend import compute_trend_features
from ultra_signals.features.momentum import compute_momentum_features
from ultra_signals.features.volatility import compute_volatility_features

def create_test_data(periods=50):
    """Create synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=periods, freq='1min')
    
    # Generate realistic price data
    base_price = 100.0
    price_changes = np.random.normal(0, 0.01, periods)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV with some volatility
    ohlcv_data = []
    for i, price in enumerate(prices):
        volatility = abs(np.random.normal(0, 0.005))
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = np.random.uniform(1000, 10000)
        
        ohlcv_data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(ohlcv_data, index=dates)

def test_features():
    """Test all feature calculations."""
    print("Creating test data...")
    ohlcv = create_test_data(100)  # 100 periods should be enough for all indicators
    
    print(f"Test data shape: {ohlcv.shape}")
    print(f"Price range: {ohlcv['close'].min():.2f} - {ohlcv['close'].max():.2f}")
    
    print("\n=== Testing Trend Features ===")
    try:
        trend_features = compute_trend_features(ohlcv, ema_short=10, ema_medium=20, ema_long=50, adx_period=14)
        print("Trend features calculated successfully:")
        for key, value in trend_features.items():
            status = "✓ Valid" if not pd.isna(value) else "✗ NaN"
            print(f"  {key}: {value:.4f} {status}")
    except Exception as e:
        print(f"Error in trend features: {e}")
    
    print("\n=== Testing Momentum Features ===")
    try:
        momentum_features = compute_momentum_features(ohlcv, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
        print("Momentum features calculated successfully:")
        for key, value in momentum_features.items():
            status = "✓ Valid" if not pd.isna(value) else "✗ NaN"
            print(f"  {key}: {value:.4f} {status}")
    except Exception as e:
        print(f"Error in momentum features: {e}")
    
    print("\n=== Testing Volatility Features ===")
    try:
        volatility_features = compute_volatility_features(ohlcv, atr_period=14, bbands_period=20, bbands_stddev=2, atr_percentile_window=50)
        print("Volatility features calculated successfully:")
        for key, value in volatility_features.items():
            status = "✓ Valid" if not pd.isna(value) else "✗ NaN"
            print(f"  {key}: {value:.4f} {status}")
    except Exception as e:
        print(f"Error in volatility features: {e}")
    
    print("\n=== Summary ===")
    print("If you see '✓ Valid' for most features, the fixes are working!")
    print("If you still see '✗ NaN' values, there may be additional issues to resolve.")

if __name__ == "__main__":
    test_features()
