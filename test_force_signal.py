#!/usr/bin/env python3
"""
Test to force a signal by creating divergent EMAs
"""

import sys
import os
import numpy as np
from typing import Dict

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_signals.engine.scoring import trend_score, momentum_score, component_scores
from ultra_signals.core.custom_types import FeatureVector, TrendFeatures, MomentumFeatures
from ultra_signals.engine.entries_exits import make_signal

def test_force_signal():
    """Test forcing a signal with divergent EMAs"""
    print("=== Testing Force Signal with Divergent EMAs ===")
    
    # Create features with divergent EMAs to force a signal
    trend_features = TrendFeatures(
        ema_short=110.0,  # Higher short EMA
        ema_medium=109.0, # Lower medium EMA  
        ema_long=108.0,   # Even lower long EMA
        adx=25.0          # Strong trend
    )
    
    momentum_features = MomentumFeatures(
        rsi=65.0,         # Bullish RSI
        macd_line=1.0,    # Positive MACD
        macd_signal=0.5,  # Positive signal
        macd_hist=0.5     # Positive histogram
    )
    
    # Create feature vector
    feature_vector = FeatureVector(
        symbol="BTCUSDT",
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
    
    # Test signal generation
    print("\n--- Testing Signal Generation ---")
    
    # Your current weights
    weights = {
        'trend': 0.5,
        'momentum': 0.5,
        'volatility': 0.0,
        'orderbook': 0.0,
        'derivatives': 0.0,
        'pullback_confluence': 0.0,
        'breakout_confluence': 0.0,
        'patterns': 0.0,
        'pullback_confluence_rs': 0.0
    }
    
    # Calculate final score
    final_score = sum(scores[comp] * weights[comp] for comp in weights.keys())
    print(f"Final Score: {final_score}")
    
    # Test with different thresholds
    thresholds = [0.001, 0.01, 0.1]
    
    # Create mock OHLCV data
    import pandas as pd
    ohlcv = pd.DataFrame({
        'close': [100.0],
        'high': [101.0], 
        'low': [99.0],
        'open': [100.0],
        'volume': [1000.0]
    })
    
    # Add ATR to features
    feature_vector.ohlcv['atr_14'] = 2.0
    
    for threshold in thresholds:
        threshold_dict = {'enter': threshold, 'exit': 0.4}
        weights_dict = {
            'trend': 0.5,
            'momentum': 0.5,
            'volatility': 0.0,
            'orderbook': 0.0,
            'derivatives': 0.0,
            'pullback_confluence': 0.0,
            'breakout_confluence': 0.0,
            'patterns': 0.0,
            'pullback_confluence_rs': 0.0
        }
        
        signal = make_signal(
            symbol="BTCUSDT",
            timeframe="5m",
            component_scores=scores,
            weights=weights_dict,
            thresholds=threshold_dict,
            features=feature_vector,
            ohlcv=ohlcv
        )
        print(f"Threshold {threshold}: {signal.decision} (score={signal.score:.3f})")

if __name__ == "__main__":
    test_force_signal()
