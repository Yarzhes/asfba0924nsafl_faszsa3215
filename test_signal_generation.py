#!/usr/bin/env python3
"""
Test signal generation with current settings
"""

import sys
import os
import numpy as np
from typing import Dict

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_signals.engine.scoring import component_scores
from ultra_signals.core.custom_types import FeatureVector, TrendFeatures, MomentumFeatures
from ultra_signals.engine.entries_exits import make_signal
import pandas as pd

def test_signal_generation():
    """Test signal generation with various market conditions"""
    print("=== Testing Signal Generation ===")
    
    # Your current settings
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
    
    thresholds = {
        'enter': 0.001,  # Your current threshold
        'exit': 0.4
    }
    
    config_params = {
        "trend": {"ema_short": 10, "ema_medium": 20, "ema_long": 50},
        "momentum": {"rsi_period": 14}
    }
    
    # Test cases based on your logs
    test_cases = [
        {
            "name": "ETHUSDT 5m - Perfect Bullish",
            "trend": TrendFeatures(
                ema_short=4390.698571428571,
                ema_medium=4390.599411764705,
                ema_long=4390.547611940298,
                adx=np.nan
            ),
            "momentum": MomentumFeatures(
                rsi=50.0,
                macd_line=0.0,
                macd_signal=0.0,
                macd_hist=0.0
            )
        },
        {
            "name": "ETHUSDT 3m - Bearish",
            "trend": TrendFeatures(
                ema_short=4394.279028182701,
                ema_medium=4394.896821584459,
                ema_long=4395.253338919127,
                adx=np.nan
            ),
            "momentum": MomentumFeatures(
                rsi=50.0,
                macd_line=0.0,
                macd_signal=0.0,
                macd_hist=0.0
            )
        },
        {
            "name": "Strong Bullish RSI",
            "trend": TrendFeatures(
                ema_short=100.0,
                ema_medium=99.0,
                ema_long=98.0,
                adx=25.0
            ),
            "momentum": MomentumFeatures(
                rsi=70.0,  # Bullish RSI
                macd_line=0.5,
                macd_signal=0.3,
                macd_hist=0.2
            )
        },
        {
            "name": "Strong Bearish RSI",
            "trend": TrendFeatures(
                ema_short=98.0,
                ema_medium=99.0,
                ema_long=100.0,
                adx=25.0
            ),
            "momentum": MomentumFeatures(
                rsi=30.0,  # Bearish RSI
                macd_line=-0.5,
                macd_signal=-0.3,
                macd_hist=-0.2
            )
        }
    ]
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        
        # Create feature vector
        feature_vector = FeatureVector(
            symbol="TEST",
            timeframe="5m",
            trend=case['trend'],
            momentum=case['momentum'],
            ohlcv={}
        )
        
        # Calculate component scores
        scores = component_scores(feature_vector, config_params)
        print(f"Component Scores: {scores}")
        
        # Calculate final score
        final_score = sum(scores.get(comp, 0.0) * w for comp, w in weights.items())
        final_score = max(-1.0, min(1.0, final_score))
        print(f"Final Score: {final_score:.4f}")
        
        # Check if signal would be generated
        if final_score >= thresholds['enter']:
            decision = "LONG"
        elif final_score <= -thresholds['enter']:
            decision = "SHORT"
        else:
            decision = "NO_TRADE"
            
        print(f"Decision: {decision}")
        print(f"Would Signal: {'YES' if decision != 'NO_TRADE' else 'NO'}")
        
        # Create mock OHLCV for signal generation
        ohlcv = pd.DataFrame({
            'close': [100.0],
            'high': [101.0],
            'low': [99.0],
            'open': [100.0],
            'volume': [1000.0]
        })
        
        try:
            signal = make_signal(
                symbol="TEST",
                timeframe="5m",
                component_scores=scores,
                weights=weights,
                thresholds=thresholds,
                features=feature_vector,
                ohlcv=ohlcv
            )
            print(f"Signal Generated: {signal.decision} (score={signal.score:.4f})")
        except Exception as e:
            print(f"Signal generation error: {e}")

if __name__ == "__main__":
    test_signal_generation()
