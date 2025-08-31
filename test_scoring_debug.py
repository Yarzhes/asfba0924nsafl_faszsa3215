#!/usr/bin/env python3
"""
Test script to debug scoring with actual feature structures
"""

import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_signals.engine.scoring import trend_score, momentum_score, component_scores
from ultra_signals.core.custom_types import FeatureVector, TrendFeatures, MomentumFeatures

def test_scoring_with_dataclasses():
    """Test scoring with actual dataclass structures"""
    print("=== Testing Scoring with Dataclasses ===")
    
    # Create test features using the actual dataclass structure
    trend_features = TrendFeatures(
        ema_short=4390.698571428571,
        ema_medium=4390.599411764705,
        ema_long=4390.547611940298,
        adx=np.nan
    )
    
    momentum_features = MomentumFeatures(
        rsi=50.0,
        macd_line=0.0,
        macd_signal=0.0,
        macd_hist=0.0
    )
    
    # Create feature vector
    feature_vector = FeatureVector(
        symbol="ETHUSDT",
        timeframe="5m",
        trend=trend_features,
        momentum=momentum_features,
        ohlcv={}
    )
    
    # Test parameters
    config_params = {
        "trend": {"ema_short": 10, "ema_medium": 20, "ema_long": 50},
        "momentum": {"rsi_period": 14}
    }
    
    print(f"Feature Vector: {feature_vector}")
    print(f"Trend Features: {trend_features}")
    print(f"Momentum Features: {momentum_features}")
    
    # Test individual scoring functions
    print("\n--- Testing Individual Scoring Functions ---")
    
    trend_result = trend_score(trend_features, config_params)
    print(f"Trend Score: {trend_result}")
    
    momentum_result = momentum_score(momentum_features, config_params)
    print(f"Momentum Score: {momentum_result}")
    
    # Test component scores
    print("\n--- Testing Component Scores ---")
    scores = component_scores(feature_vector, config_params)
    print(f"Component Scores: {scores}")
    
    # Test with different EMA configurations
    print("\n--- Testing Different EMA Configurations ---")
    
    test_cases = [
        {
            "name": "Perfect Bullish",
            "trend": TrendFeatures(ema_short=100.0, ema_medium=99.0, ema_long=98.0, adx=25.0)
        },
        {
            "name": "Perfect Bearish", 
            "trend": TrendFeatures(ema_short=98.0, ema_medium=99.0, ema_long=100.0, adx=25.0)
        },
        {
            "name": "Mixed",
            "trend": TrendFeatures(ema_short=100.0, ema_medium=98.0, ema_long=99.0, adx=25.0)
        }
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        trend_score_result = trend_score(case['trend'], config_params)
        print(f"  Trend Score: {trend_score_result:.3f}")
        
        # Create feature vector for this case
        fv = FeatureVector(
            symbol="TEST",
            timeframe="5m", 
            trend=case['trend'],
            momentum=momentum_features,
            ohlcv={}
        )
        
        comp_scores = component_scores(fv, config_params)
        print(f"  Component Scores: {comp_scores}")

def test_scoring_with_dict():
    """Test scoring with dictionary structure (fallback)"""
    print("\n=== Testing Scoring with Dictionary Structure ===")
    
    # Test with dictionary structure (like from logs)
    trend_dict = {
        "ema_short": 4390.698571428571,
        "ema_medium": 4390.599411764705,
        "ema_long": 4390.547611940298,
        "adx": np.nan
    }
    
    momentum_dict = {
        "rsi": 50.0,
        "macd_line": 0.0,
        "macd_signal": 0.0,
        "macd_hist": 0.0
    }
    
    config_params = {
        "trend": {"ema_short": 10, "ema_medium": 20, "ema_long": 50},
        "momentum": {"rsi_period": 14}
    }
    
    print(f"Trend Dict: {trend_dict}")
    print(f"Momentum Dict: {momentum_dict}")
    
    trend_result = trend_score(trend_dict, config_params)
    print(f"Trend Score (dict): {trend_result}")
    
    momentum_result = momentum_score(momentum_dict, config_params)
    print(f"Momentum Score (dict): {momentum_result}")

if __name__ == "__main__":
    test_scoring_with_dataclasses()
    test_scoring_with_dict()

