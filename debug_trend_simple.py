import sys
import os
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Simple test without complex imports
def test_basic():
    print("Testing basic functionality...")
    
    # Test the _get function logic directly
    test_dict = {
        'ema_10': 50000.0,
        'ema_20': 49500.0,
        'ema_50': 49000.0
    }
    
    # Simulate the _get function behavior
    def simple_get(src, attr=None, key=None, default=None):
        if src is None:
            return default
        
        if isinstance(src, dict):
            if key is not None and key in src:
                return src[key]
            if attr is not None and attr in src:
                return src[attr]
            return default
        
        if attr is not None and hasattr(src, attr):
            return getattr(src, attr)
        
        return default
    
    # Test key lookups
    ema_s = simple_get(test_dict, "ema_short", "ema_10")
    ema_m = simple_get(test_dict, "ema_medium", "ema_20")
    ema_l = simple_get(test_dict, "ema_long", "ema_50")
    
    print(f"EMA values: s={ema_s}, m={ema_m}, l={ema_l}")
    
    # Test NaN detection
    if (ema_s is None or np.isnan(ema_s) or 
        ema_m is None or np.isnan(ema_m) or 
        ema_l is None or np.isnan(ema_l)):
        print("NaN/None detected!")
        return 0.0
    
    # Test scoring logic
    ema_s = float(ema_s)
    ema_m = float(ema_m)
    ema_l = float(ema_l)
    
    print(f"Converted EMAs: s={ema_s:.6f}, m={ema_m:.6f}, l={ema_l:.6f}")
    print(f"s>m: {ema_s > ema_m}, m>l: {ema_m > ema_l}")
    print(f"Perfect bull: {ema_s > ema_m > ema_l}")
    
    if ema_s > ema_m > ema_l:
        print("Should return +1.0 (perfect bullish)")
        return 1.0
    
    print("Should calculate partial score...")
    score = 0.0
    if ema_s > ema_m:
        score += 0.5
    if ema_m > ema_l:
        score += 0.5
    
    print(f"Final score: {score}")
    return score

if __name__ == "__main__":
    result = test_basic()
    print(f"Test result: {result}")
