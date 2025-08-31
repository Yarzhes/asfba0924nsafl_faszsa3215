#!/usr/bin/env python3
"""
Force Signal Test - Create divergent EMAs to generate a signal immediately
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_signals.engine.scoring import trend_score, momentum_score, component_scores
from ultra_signals.core.custom_types import FeatureVector, TrendFeatures, MomentumFeatures
from ultra_signals.engine.entries_exits import make_signal

def test_force_signal_quick():
    """Test forcing a signal with divergent EMAs"""
    print("=== Force Signal Test - Quick ===")
    
    # Create features with DIVERGENT EMAs to force a strong signal
    trend_features = TrendFeatures(
        ema_short=110.0,  # Much higher short EMA
        ema_medium=105.0, # Lower medium EMA  
        ema_long=100.0,   # Much lower long EMA
        adx=30.0          # Strong trend
    )
    
    momentum_features = MomentumFeatures(
        rsi=70.0,         # Very bullish RSI
        macd_line=2.0,    # Strong positive MACD
        macd_signal=1.0,  # Positive signal
        macd_hist=1.0     # Strong positive histogram
    )
    
    # Create feature vector
    feature_vector = FeatureVector(
        symbol="BTCUSDT",
        timeframe="5m",
        trend=trend_features,
        momentum=momentum_features,
        ohlcv={'atr_14': 5.0}  # Add ATR for signal generation
    )
    
    # Test parameters (using your current settings)
    config_params = {
        "trend": {"ema_short": 3, "ema_medium": 5, "ema_long": 10},
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
    
    # Calculate final score with your weights
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
    
    final_score = sum(scores[comp] * weights[comp] for comp in weights.keys())
    print(f"Final Score: {final_score:.4f}")
    
    # Generate signal with your threshold
    thresholds = {'enter': 0.001, 'exit': 0.4}
    
    # Create mock OHLCV data
    ohlcv = pd.DataFrame({
        'close': [107.0],
        'high': [108.0], 
        'low': [106.0],
        'open': [106.5],
        'volume': [1000.0]
    })
    
    signal = make_signal(
        symbol="BTCUSDT",
        timeframe="5m",
        component_scores=scores,
        weights=weights,
        thresholds=thresholds,
        features=feature_vector,
        ohlcv=ohlcv
    )
    
    print(f"\n🎯 Signal Result:")
    print(f"Decision: {signal.decision}")
    print(f"Score: {signal.score:.4f}")
    print(f"Confidence: {signal.confidence:.2f}%")
    
    if signal.decision != "NO_TRADE":
        print(f"Entry: {signal.entry_price:.2f}")
        print(f"Stop Loss: {signal.stop_loss:.2f}")
        print(f"Take Profit 1: {signal.take_profit_1:.2f}")
        print(f"Take Profit 2: {signal.take_profit_2:.2f}")
        print("🎉 SUCCESS! Signal generated!")
        
        # Test Telegram sending
        print("\n--- Testing Telegram ---")
        try:
            import requests
            bot_token = "8360503431:AAFpGnCkc6JbUUgq6OFo9KWJw3Kp200jSFs"
            chat_id = "7072100094"
            
            message = f"""🎯 Trading Helper Signal Test

{signal.symbol} | {signal.decision} | ENTRY:{signal.entry_price:.2f} | SL:{signal.stop_loss:.2f} | TP:{signal.take_profit_1:.2f}/{signal.take_profit_2:.2f} | Lev:10 | p:{signal.confidence/100:.2f} | regime:trend | veto:none | code:trend_breakout

Score: {signal.score:.4f}
Confidence: {signal.confidence:.1f}%

This is a test signal to verify the system is working correctly."""

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": message
            }
            
            response = requests.post(url, data=data)
            if response.status_code == 200:
                print("✅ Telegram message sent successfully!")
            else:
                print(f"❌ Telegram error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Telegram error: {e}")
    else:
        print("❌ No signal generated - score too low")

if __name__ == "__main__":
    test_force_signal_quick()

