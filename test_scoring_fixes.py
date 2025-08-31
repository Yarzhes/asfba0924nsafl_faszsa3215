#!/usr/bin/env python3
"""
Test script to debug scoring issues and verify signal generation
"""

def test_trend_scoring():
    """Test trend scoring with various EMA configurations"""
    print("=== Testing Trend Scoring ===")
    
    # Test cases from your logs
    test_cases = [
        {
            "name": "ETHUSDT 5m - Very Close EMAs",
            "ema_short": 4390.698571428571,
            "ema_medium": 4390.599411764705,
            "ema_long": 4390.547611940298
        },
        {
            "name": "ETHUSDT 3m - Slightly Bullish",
            "ema_short": 4394.279028182701,
            "ema_medium": 4394.896821584459,
            "ema_long": 4395.253338919127
        },
        {
            "name": "LINKUSDT 3m - Slightly Bullish",
            "ema_short": 23.370518302559116,
            "ema_medium": 23.373573738607327,
            "ema_long": 23.375361223879864
        },
        {
            "name": "Perfect Bullish Alignment",
            "ema_short": 100.0,
            "ema_medium": 99.0,
            "ema_long": 98.0
        },
        {
            "name": "Perfect Bearish Alignment", 
            "ema_short": 98.0,
            "ema_medium": 99.0,
            "ema_long": 100.0
        }
    ]
    
    for case in test_cases:
        ema_s = case["ema_short"]
        ema_m = case["ema_medium"] 
        ema_l = case["ema_long"]
        
        print(f"\n{case['name']}:")
        print(f"  EMAs: s={ema_s:.6f}, m={ema_m:.6f}, l={ema_l:.6f}")
        print(f"  s>m: {ema_s > ema_m}, m>l: {ema_m > ema_l}")
        
        # Calculate trend score using the same logic as the engine
        score = 0.0
        if ema_s > ema_m > ema_l:
            score = 1.0  # Perfect bullish
        elif ema_s < ema_m < ema_l:
            score = -1.0  # Perfect bearish
        else:
            # Partial scoring
            if ema_s > ema_m:
                score += 0.5
            elif ema_s < ema_m:
                score -= 0.5
                
            if ema_m > ema_l:
                score += 0.5
            elif ema_m < ema_l:
                score -= 0.5
        
        score = max(-1.0, min(1.0, score))
        print(f"  Trend Score: {score:.3f}")
        
        # Calculate momentum score (RSI = 50.0, MACD = 0.0 from logs)
        rsi = 50.0
        macd_hist = 0.0
        rsi_part = (rsi - 50.0) / 50.0
        macd_part = max(-1.0, min(1.0, macd_hist * 10.0))
        momentum_score = 0.8 * rsi_part + 0.2 * macd_part
        print(f"  Momentum Score: {momentum_score:.3f}")
        
        # Calculate final score with current weights
        weights = {'trend': 0.5, 'momentum': 0.5}
        final_score = weights['trend'] * score + weights['momentum'] * momentum_score
        print(f"  Final Score: {final_score:.3f}")
        print(f"  Would Signal: {'YES' if abs(final_score) >= 0.001 else 'NO'}")

def test_momentum_scoring():
    """Test momentum scoring with various RSI/MACD values"""
    print("\n=== Testing Momentum Scoring ===")
    
    test_cases = [
        {"rsi": 50.0, "macd_hist": 0.0, "name": "Neutral"},
        {"rsi": 60.0, "macd_hist": 0.1, "name": "Slightly Bullish"},
        {"rsi": 40.0, "macd_hist": -0.1, "name": "Slightly Bearish"},
        {"rsi": 70.0, "macd_hist": 0.2, "name": "Bullish"},
        {"rsi": 30.0, "macd_hist": -0.2, "name": "Bearish"},
    ]
    
    for case in test_cases:
        rsi = case["rsi"]
        macd_hist = case["macd_hist"]
        
        rsi_part = (rsi - 50.0) / 50.0
        macd_part = max(-1.0, min(1.0, macd_hist * 10.0))
        momentum_score = 0.8 * rsi_part + 0.2 * macd_part
        
        print(f"{case['name']}: RSI={rsi}, MACD={macd_hist:.2f} → Score={momentum_score:.3f}")

if __name__ == "__main__":
    test_trend_scoring()
    test_momentum_scoring()
