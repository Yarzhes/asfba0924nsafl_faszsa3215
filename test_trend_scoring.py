#!/usr/bin/env python3
"""Quick test to debug trend scoring function"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultra_signals.engine.scoring import trend_score
from loguru import logger
import logging

# Configure logging to see INFO level messages
logger.remove()
logger.add(sys.stderr, level="INFO")

def test_trend_scoring():
    """Test trend scoring with sample data"""
    
    # Test case 1: Dict with clear bullish trend (short > medium > long)
    test_data_1 = {
        'ema_10': 50000.0,
        'ema_20': 49500.0,
        'ema_50': 49000.0
    }
    
    params = {
        'trend': {
            'ema_short': 10,
            'ema_medium': 20,
            'ema_long': 50
        }
    }
    
    print("=" * 60)
    print("TEST 1: Clear bullish trend (50000 > 49500 > 49000)")
    print("=" * 60)
    score_1 = trend_score(test_data_1, params)
    print(f"Result: {score_1}")
    
    # Test case 2: Dict with clear bearish trend (short < medium < long)
    test_data_2 = {
        'ema_10': 49000.0,
        'ema_20': 49500.0,
        'ema_50': 50000.0
    }
    
    print("\n" + "=" * 60)
    print("TEST 2: Clear bearish trend (49000 < 49500 < 50000)")
    print("=" * 60)
    score_2 = trend_score(test_data_2, params)
    print(f"Result: {score_2}")
    
    # Test case 3: Dict with mixed signals
    test_data_3 = {
        'ema_10': 49750.0,
        'ema_20': 49500.0,
        'ema_50': 50000.0
    }
    
    print("\n" + "=" * 60)
    print("TEST 3: Mixed signals (49750 > 49500 < 50000)")
    print("=" * 60)
    score_3 = trend_score(test_data_3, params)
    print(f"Result: {score_3}")
    
    # Test case 4: Dict with NaN values
    test_data_4 = {
        'ema_10': float('nan'),
        'ema_20': 49500.0,
        'ema_50': 50000.0
    }
    
    print("\n" + "=" * 60)
    print("TEST 4: NaN values (nan, 49500, 50000)")
    print("=" * 60)
    score_4 = trend_score(test_data_4, params)
    print(f"Result: {score_4}")

if __name__ == "__main__":
    test_trend_scoring()
