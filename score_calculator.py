#!/usr/bin/env python3
"""
Quick calculator to check final scores from component scores
"""

def calculate_final_scores():
    print("=== Final Score Calculator ===")
    
    # Weights from settings.yaml
    weights = {
        'trend': 0.5,
        'momentum': 0.5,
        'volatility': 0.0,
        'orderbook': 0.0,
        'derivatives': 0.0,
        'pullback_confluence': 0.0,
        'breakout_confluence': 0.0,
        'pullback_confluence_rs': 0.0
    }
    
    # Example component scores from logs
    examples = [
        {'trend': -1.0, 'momentum': 0.23296357724952088, 'volatility': 0.0, 'orderbook': 0.0, 'derivatives': 0.0, 'pullback_confluence': 0.0, 'breakout_confluence': 0.0, 'pullback_confluence_rs': 0.0},
        {'trend': 0.0, 'momentum': 0.3537333137459905, 'volatility': 0.0, 'orderbook': 0.0, 'derivatives': 0.0, 'pullback_confluence': 0.0, 'breakout_confluence': 0.0, 'pullback_confluence_rs': 0.0},
        {'trend': 0.0, 'momentum': -0.2877434593483638, 'volatility': 0.0, 'orderbook': 0.0, 'derivatives': 0.0, 'pullback_confluence': 0.0, 'breakout_confluence': 0.0, 'pullback_confluence_rs': 0.0},
        {'trend': -1.0, 'momentum': -0.29141808377866085, 'volatility': 0.0, 'orderbook': 0.0, 'derivatives': 0.0, 'pullback_confluence': 0.0, 'breakout_confluence': 0.0, 'pullback_confluence_rs': 0.0},
    ]
    
    threshold_enter = 0.1
    
    print(f"Weights: {weights}")
    print(f"Entry threshold: ±{threshold_enter}")
    print()
    
    for i, scores in enumerate(examples, 1):
        # Calculate final score
        final_score = sum(scores.get(comp, 0.0) * weight for comp, weight in weights.items())
        final_score = max(-1.0, min(1.0, final_score))  # Clip to [-1, 1]
        
        # Check decision
        if final_score >= threshold_enter:
            decision = "LONG"
        elif final_score <= -threshold_enter:
            decision = "SHORT"
        else:
            decision = "NO_TRADE"
            
        print(f"Example {i}:")
        print(f"  Trend: {scores['trend']:.3f} * {weights['trend']} = {scores['trend'] * weights['trend']:.3f}")
        print(f"  Momentum: {scores['momentum']:.3f} * {weights['momentum']} = {scores['momentum'] * weights['momentum']:.3f}")
        print(f"  Final Score: {final_score:.3f}")
        print(f"  Decision: {decision}")
        print()

if __name__ == "__main__":
    calculate_final_scores()
