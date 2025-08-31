#!/usr/bin/env python3
"""
Ultra Sensitive Signal Test - Test with minimum requirements
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

def test_ultra_sensitive():
    """Test with ultra-sensitive settings"""
    print("=== Ultra Sensitive Signal Test ===")
    
    # Test with VERY small EMA differences (like your live data)
    trend_features = TrendFeatures(
        ema_short=100.01,  # Tiny difference
        ema_medium=100.00, # Same as long
        ema_long=100.00,   # Same as medium
        adx=5.0            # Very low ADX (sideways market)
    )
    
    momentum_features = MomentumFeatures(
        rsi=51.0,          # Slightly above neutral
        macd_line=0.01,    # Tiny positive MACD
        macd_signal=0.0,   # Neutral signal
        macd_hist=0.01     # Tiny positive histogram
    )
    
    # Create feature vector
    feature_vector = FeatureVector(
        symbol="BTCUSDT",
        timeframe="5m",
        trend=trend_features,
        momentum=momentum_features,
        ohlcv={'atr_14': 1.0}  # Small ATR
    )
    
    # Test parameters (using your ultra-sensitive settings)
    config_params = {
        "trend": {"ema_short": 2, "ema_medium": 3, "ema_long": 5},
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
    print(f"Final Score: {final_score:.6f}")
    
    # Test with ultra-low threshold
    thresholds = {'enter': 0.0001, 'exit': 0.1}  # Your ultra-sensitive settings
    
    # Create mock OHLCV data
    ohlcv = pd.DataFrame({
        'close': [100.0],
        'high': [100.1], 
        'low': [99.9],
        'open': [100.0],
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
    print(f"Score: {signal.score:.6f}")
    print(f"Confidence: {signal.confidence:.2f}%")
    
    if signal.decision != "NO_TRADE":
        print(f"Entry: {signal.entry_price:.2f}")
        print(f"Stop Loss: {signal.stop_loss:.2f}")
        print(f"Take Profit 1: {signal.take_profit_1:.2f}")
        print(f"Take Profit 2: {signal.take_profit_2:.2f}")
        print("🎉 SUCCESS! Signal generated with ultra-sensitive settings!")
        
        # Test Telegram sending
        print("\n--- Testing Telegram ---")
        try:
            import requests
            bot_token = "8360503431:AAFpGnCkc6JbUUgq6OFo9KWJw3Kp200jSFs"
            chat_id = "7072100094"
            
            message = f"""🎯 Ultra Sensitive Signal Test

{signal.symbol} | {signal.decision} | ENTRY:{signal.entry_price:.2f} | SL:{signal.stop_loss:.2f} | TP:{signal.take_profit_1:.2f}/{signal.take_profit_2:.2f} | Lev:10 | p:{signal.confidence/100:.2f} | regime:trend | veto:none | code:trend_breakout

Score: {signal.score:.6f}
Confidence: {signal.confidence:.1f}%

This is an ultra-sensitive test signal."""

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
        print("❌ No signal generated - even with ultra-sensitive settings")
        print("This means the market is completely sideways with no trend at all")

if __name__ == "__main__":
    test_ultra_sensitive()

