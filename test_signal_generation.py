"""
Quick test script to verify signal generation works without any warmup delays
"""
import yaml
from ultra_signals.engine.scoring import component_scores
from ultra_signals.engine.entries_exits import make_signal
from ultra_signals.core.custom_types import FeatureVector
import pandas as pd
import numpy as np

# Load settings
with open('settings.yaml', 'r') as f:
    settings = yaml.safe_load(f)

print("=== SIGNAL GENERATION TEST ===")
print(f"Entry threshold: {settings['engine']['thresholds']['enter']}")
print(f"Scoring weights: {settings['engine']['scoring_weights']}")
print()

# Create fake feature vector with some values
feature_vector = FeatureVector(
    symbol="BTCUSDT",
    timeframe="5m",
    ohlcv={
        "ema_short": 50000.0,
        "ema_medium": 49900.0, 
        "ema_long": 49800.0,
        "rsi": 45.0,
        "macd": 0.1,
        "atr_3": 100.0
    },
    orderbook={},
    derivatives={},
    funding={}
)

# Test component scoring
print("Testing component scoring...")
try:
    scores = component_scores(feature_vector, settings['features'])
    print(f"✅ Component scores: {scores}")
except Exception as e:
    print(f"❌ Component scoring failed: {e}")
    exit(1)

# Test signal generation
print("\nTesting signal generation...")
try:
    # Create fake OHLCV dataframe
    ohlcv = pd.DataFrame({
        'close': [50000.0, 50100.0, 50050.0],
        'high': [50200.0, 50300.0, 50250.0],
        'low': [49800.0, 49900.0, 49850.0],
        'open': [49900.0, 50000.0, 50100.0],
        'volume': [1000, 1100, 1050]
    })
    
    # Create simple threshold object
    class Thresholds:
        def __init__(self, enter, exit):
            self.enter = enter
            self.exit = exit
    
    thresholds = Thresholds(
        enter=settings['engine']['thresholds']['enter'],
        exit=settings['engine']['thresholds']['exit']
    )
    
    signal = make_signal(
        symbol="BTCUSDT",
        timeframe="5m", 
        component_scores=scores,
        weights=settings['engine']['scoring_weights'],
        thresholds=thresholds,
        features=feature_vector,
        ohlcv=ohlcv
    )
    
    print(f"✅ Signal generated successfully!")
    print(f"   Decision: {signal.decision}")
    print(f"   Score: {signal.score:.3f}")
    print(f"   Confidence: {signal.confidence:.1f}")
    print(f"   Entry price: {signal.entry_price}")
    
    if signal.decision != "NO_TRADE":
        print(f"   🎉 SIGNAL FOUND! {signal.decision} at {signal.entry_price}")
    else:
        print(f"   ⚠️  No trade (score {signal.score:.3f} vs threshold {settings['engine']['thresholds']['enter']})")
        
except Exception as e:
    print(f"❌ Signal generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== TEST COMPLETE ===")
print("✅ Signal generation system is working!")
