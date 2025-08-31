"""
Signal Generation Logic

This module contains the logic for converting a final, weighted score into a
concrete trading signal (`LONG`, `SHORT`, or `NO_TRADE`). It defines the
entry/exit decision-making process and calculates key parameters for the signal,
such as confidence, entry price, stop-loss, and take-profit levels.

Design Principles:
- Threshold-Based: Decisions are made by comparing the final score against
  configurable thresholds from the settings file.
- Clear Signal Object: The output is a well-defined `Signal` object that
  encapsulates all necessary information for downstream consumers (like
  the transport/notification module).
- Risk-Aware Calculations: Stop-loss and take-profit levels are calculated
  based on the current market volatility (ATR), providing a dynamic and
  adaptive approach to risk management.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal

import numpy as np

from ultra_signals.core.custom_types import FeatureVector


# --- Signal Data Model ---

@dataclass
class Signal:
    """
    Represents a trading signal.
    This object is the final output of the engine for a given kline event.
    """
    # Fields without default values
    symbol: str
    timeframe: str
    decision: Literal["LONG", "SHORT", "NO_TRADE"]
    feature_vector: FeatureVector
    
    # Fields with default values
    score: float = 0.0
    confidence: float = 0.0 # From 0 to 1
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    component_scores: Dict[str, float] = field(default_factory=dict)
    vote_detail: Dict = field(default_factory=dict)  # Added for risk filter compatibility


# --- Signal Generation Logic ---

def _calculate_confidence(score: float, alpha: float = 2.0) -> float:
    """
    Maps a score from [-1, 1] to a confidence level [0, 1]
    using a scaled sigmoid-like function.
    """
    # Simple sigmoid: 1 / (1 + exp(-ax))
    # We use abs(score) because confidence is about magnitude, not direction.
    return 1.0 / (1 + np.exp(-alpha * abs(score) * 5)) # Scaled to feel responsive

def make_signal(
    symbol: str,
    timeframe: str,
    component_scores: Dict[str, float],
    weights: Dict[str, float],
    thresholds: Dict[str, float],
    features: FeatureVector,
    ohlcv: "pd.DataFrame",
) -> Signal:
    """
    Creates a Signal object based on the weighted score of feature components.

    Args:
        symbol: The trading symbol.
        timeframe: The kline timeframe.
        component_scores: A dictionary of scores (e.g., {'trend': 0.8}).
        weights: A dictionary of weights for each score component.
        thresholds: A dictionary with 'enter' and 'exit' thresholds.
        features: The raw `FeatureVector` for context.
        ohlcv: The OHLCV dataframe to get the latest close and ATR.

    Returns:
        A `Signal` object with a decision and all relevant details.
    """
    # 1. Calculate the final weighted score
    final_score = sum(
        component_scores.get(comp, 0.0) * w for comp, w in weights.items()
    )
    final_score = np.clip(final_score, -1.0, 1.0) # Ensure it's in [-1, 1]

    # 2. Make a decision based on the entry threshold
    decision: Literal["LONG", "SHORT", "NO_TRADE"] = "NO_TRADE"
    # Handle both object and dictionary thresholds
    enter_threshold = thresholds.enter if hasattr(thresholds, 'enter') else thresholds.get('enter', 0.01)
    
    if final_score >= enter_threshold:
        decision = "LONG"
    elif final_score <= -enter_threshold:
        decision = "SHORT"

    # 3. If no trade, return a simple signal object
    if decision == "NO_TRADE":
        return Signal(
            symbol=symbol,
            timeframe=timeframe,
            decision="NO_TRADE",
            score=final_score,
            feature_vector=features,
            component_scores=component_scores,
        )

    # 4. For LONG/SHORT signals, calculate confidence and price levels
    confidence = _calculate_confidence(final_score)
    last_close = ohlcv['close'].iloc[-1]
    
    # Find ATR from features (must match the key from the volatility module)
    # Find ATR from features (must match the key from the volatility module)
    ohlcv_features = features.ohlcv
    atr_key = next((k for k in ohlcv_features if k.startswith('atr_')), None)
    if not atr_key or np.isnan(ohlcv_features[atr_key]):
        # Fallback if ATR is not available, though this should be rare after warmup
        atr = (ohlcv['high'] - ohlcv['low']).iloc[-1] * 0.1
    else:
        atr = ohlcv_features[atr_key]

    # 5. Calculate SL and TP levels based on ATR
    if decision == "LONG":
        stop_loss = last_close - (1.4 * atr)
        take_profit_1 = last_close + (1.0 * atr)
        take_profit_2 = last_close + (2.0 * atr)
    else: # SHORT
        stop_loss = last_close + (1.4 * atr)
        take_profit_1 = last_close - (1.0 * atr)
        take_profit_2 = last_close - (2.0 * atr)
        
    return Signal(
        symbol=symbol,
        timeframe=timeframe,
        decision=decision,
        score=final_score,
        confidence=confidence,
        entry_price=last_close,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        component_scores=component_scores,
        feature_vector=features
    )