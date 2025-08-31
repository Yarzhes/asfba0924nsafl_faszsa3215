"""
Position Sizing Module

This module provides position sizing functionality including:
- Dynamic position sizing based on confidence and volatility
- Kelly-lite sizing for optimal risk allocation
- ATR-based position sizing
- Volatility scaling adjustments
"""

from ultra_signals.engine.position_sizer import (
    PositionSizer,
    PositionSizeResult,
    determine_position_size,
    apply_volatility_scaling,
    kelly_lite_multiplier,
    kelly_lite_size,
    atr_position_size
)

__all__ = [
    'PositionSizer',
    'PositionSizeResult', 
    'determine_position_size',
    'apply_volatility_scaling',
    'kelly_lite_multiplier',
    'kelly_lite_size',
    'atr_position_size'
]
