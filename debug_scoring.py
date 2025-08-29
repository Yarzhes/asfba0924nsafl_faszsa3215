#!/usr/bin/env python3
"""
Debug script to check scoring and signal generation
"""
import os
import sys
# Add current directory to path to find ultra_signals module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_signals.core.config import Settings, load_settings
from ultra_signals.data.feed import DataProvider
from ultra_signals.features.storage import FeatureStore
from ultra_signals.engine.scoring import component_scores
from ultra_signals.engine.entries_exits import make_signal
from ultra_signals.risk.filters import apply_filters
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_scoring_pipeline():
    """Test the scoring pipeline with current data"""
    print("=== Debug Scoring Pipeline ===")
    
    # Load settings
    settings = load_settings("settings.yaml")
    print(f"Entry threshold: {settings.engine.thresholds.enter}")
    print(f"Scoring weights: {settings.engine.scoring_weights}")
    
    # Initialize data provider and feature store
    data_provider = DataProvider(settings.data)
    feature_store = FeatureStore(data_provider, settings.features)
    
    # Get latest data for a few symbols
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    for symbol in test_symbols:
        print(f"\n--- Testing {symbol} ---")
        
        try:
            # Get latest kline
            klines = data_provider.get_recent_klines(symbol, "1m", limit=100)
            if klines.empty:
                print(f"No data for {symbol}")
                continue
                
            latest_kline = klines.iloc[-1]
            print(f"Latest price: {latest_kline['close']}")
            
            # Generate features
            feature_vector = feature_store.get_features(symbol, "1m")
            if not feature_vector:
                print(f"No features generated for {symbol}")
                continue
            
            # Get component scores
            scores = component_scores(feature_vector, settings.model_dump())
            print(f"Component scores: {scores}")
            
            # Calculate final score
            weights = settings.engine.scoring_weights
            final_score = sum(scores[k] * getattr(weights, k, 0.0) for k in scores.keys())
            print(f"Final score: {final_score}")
            
            # Test signal generation
            signal = make_signal(
                symbol=symbol,
                timeframe="1m",
                component_scores=scores,
                weights=weights.model_dump(),
                thresholds=settings.engine.thresholds.model_dump(),
                features=feature_vector,
                ohlcv=klines
            )
            print(f"Signal generated: {signal.direction if signal else 'None'}")
            
            if signal:
                # Test risk filtering
                risk_result = apply_filters(signal, feature_store, settings.model_dump())
                print(f"Risk filter passed: {risk_result.passed}")
                if not risk_result.passed:
                    print(f"Risk filter reason: {risk_result.reason}")
            
        except Exception as e:
            print(f"Error testing {symbol}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_scoring_pipeline()
