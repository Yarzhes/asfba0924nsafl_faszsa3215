#!/usr/bin/env python3
"""
Quick Signal Test - Simulate conditions to generate a signal immediately
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_signals.core.config import load_settings
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.engine.scoring import component_scores
from ultra_signals.engine.entries_exits import make_signal
from ultra_signals.core.custom_types import FeatureVector, TrendFeatures, MomentumFeatures

def test_quick_signal():
    """Test generating a signal with minimal warmup"""
    print("=== Quick Signal Test ===")
    
    # Load settings
    settings = load_settings('settings.yaml')
    print(f"Warmup periods: {settings.features.warmup_periods}")
    print(f"EMA periods: {settings.features.trend.ema_short}, {settings.features.trend.ema_medium}, {settings.features.trend.ema_long}")
    
    # Create feature store with minimal warmup
    feature_store = FeatureStore(warmup_periods=2, settings=settings.model_dump())
    
    # Create mock OHLCV data with enough bars for EMA calculation
    # We need at least 10 bars for the longest EMA (10 periods)
    ohlcv_data = []
    base_price = 100.0
    
    # Create trending data (increasing prices)
    for i in range(15):  # More than enough bars
        price = base_price + (i * 0.5)  # Trending upward
        ohlcv_data.append({
            'ts': pd.Timestamp.now() - pd.Timedelta(minutes=5*(15-i)),
            'open': price - 0.1,
            'high': price + 0.2,
            'low': price - 0.2,
            'close': price,
            'volume': 1000.0
        })
    
    ohlcv_df = pd.DataFrame(ohlcv_data)
    ohlcv_df.set_index('ts', inplace=True)
    
    print(f"Created {len(ohlcv_df)} bars of trending data")
    print(f"Price range: {ohlcv_df['close'].min():.2f} - {ohlcv_df['close'].max():.2f}")
    
    # Feed data to feature store
    symbol = "BTCUSDT"
    timeframe = "5m"
    
    for idx, row in ohlcv_df.iterrows():
        # Create a bar with timestamp
        bar_with_timestamp = row.to_frame().T
        bar_with_timestamp["timestamp"] = idx
        feature_store.on_bar(symbol, timeframe, bar_with_timestamp)
    
    # Get latest features
    latest_features = feature_store.get_latest_features(symbol, timeframe)
    
    if latest_features:
        print(f"✅ Features computed successfully!")
        print(f"Trend features: {latest_features.get('trend', 'Not found')}")
        print(f"Momentum features: {latest_features.get('momentum', 'Not found')}")
        
        # Create feature vector
        feature_vector = FeatureVector(
            symbol=symbol,
            timeframe=timeframe,
            trend=latest_features.get('trend'),
            momentum=latest_features.get('momentum'),
            ohlcv=latest_features.get('ohlcv', {})
        )
        
        # Calculate component scores
        config_params = {
            "trend": {"ema_short": 3, "ema_medium": 5, "ema_long": 10},
            "momentum": {"rsi_period": 14}
        }
        
        scores = component_scores(feature_vector, config_params)
        print(f"Component scores: {scores}")
        
        # Calculate final score
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
        
        final_score = sum(scores.get(comp, 0.0) * w for comp, w in weights.items())
        print(f"Final score: {final_score:.4f}")
        
        # Generate signal
        thresholds = {'enter': 0.001, 'exit': 0.4}
        
        signal = make_signal(
            symbol=symbol,
            timeframe=timeframe,
            component_scores=scores,
            weights=weights,
            thresholds=thresholds,
            features=feature_vector,
            ohlcv=ohlcv_df
        )
        
        print(f"🎯 Signal generated: {signal.decision}")
        print(f"Score: {signal.score:.4f}")
        print(f"Confidence: {signal.confidence:.2f}%")
        
        if signal.decision != "NO_TRADE":
            print(f"Entry: {signal.entry_price:.2f}")
            print(f"Stop Loss: {signal.stop_loss:.2f}")
            print(f"Take Profit 1: {signal.take_profit_1:.2f}")
            print(f"Take Profit 2: {signal.take_profit_2:.2f}")
            print("🎉 SUCCESS! Signal generated!")
        else:
            print("❌ No signal generated - score too low")
    else:
        print("❌ No features computed")

if __name__ == "__main__":
    test_quick_signal()
