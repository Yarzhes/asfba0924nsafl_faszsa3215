"""
Triple-Barrier Method for Trade Outcome Labeling

This module implements the triple-barrier method for labeling trade outcomes
in backtesting. The method defines three barriers: take profit (TP), stop loss (SL),
and time horizon. The first barrier hit determines the outcome.

Reference: Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class TripleBarrierResult:
    """Result of triple-barrier labeling."""
    outcome: int  # +1 for win, -1 for loss, 0 for timeout
    hit_barrier: str  # 'tp', 'sl', or 'timeout'
    horizon_bars: int  # Number of bars until barrier hit
    exit_price: float  # Price at barrier hit
    return_pct: float  # Percentage return


def label_trades(
    prices: pd.Series,
    entry_idx: int,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_horizon_bars: int = 20
) -> TripleBarrierResult:
    """
    Label a trade using the triple-barrier method.
    
    Args:
        prices: Price series (OHLCV close prices)
        entry_idx: Index of entry bar
        pt_mult: Take profit multiplier (risk-reward ratio)
        sl_mult: Stop loss multiplier (ATR multiplier)
        max_horizon_bars: Maximum bars to wait for outcome
        
    Returns:
        TripleBarrierResult with outcome details
    """
    if entry_idx >= len(prices) - 1:
        return TripleBarrierResult(
            outcome=0,
            hit_barrier='timeout',
            horizon_bars=0,
            exit_price=prices.iloc[-1],
            return_pct=0.0
        )
    
    entry_price = prices.iloc[entry_idx]
    entry_time = prices.index[entry_idx]
    
    # Calculate barriers
    tp_price = entry_price * (1 + pt_mult * 0.01)  # 1% base risk
    sl_price = entry_price * (1 - sl_mult * 0.01)  # 1% base risk
    
    logger.debug(f"[LABELING] Entry: {entry_price}, TP: {tp_price}, SL: {sl_price}")
    
    # Look forward from entry
    future_prices = prices.iloc[entry_idx + 1:entry_idx + max_horizon_bars + 1]
    
    if len(future_prices) == 0:
        return TripleBarrierResult(
            outcome=0,
            hit_barrier='timeout',
            horizon_bars=0,
            exit_price=entry_price,
            return_pct=0.0
        )
    
    # Find first barrier hit
    tp_hit = future_prices >= tp_price
    sl_hit = future_prices <= sl_price
    
    if tp_hit.any():
        tp_bar = tp_hit.argmax() + 1  # +1 because we start from entry_idx + 1
    else:
        tp_bar = max_horizon_bars + 1
    
    if sl_hit.any():
        sl_bar = sl_hit.argmax() + 1  # +1 because we start from entry_idx + 1
    else:
        sl_bar = max_horizon_bars + 1
    
    # Determine outcome
    if tp_bar <= sl_bar and tp_bar <= max_horizon_bars:
        # Take profit hit first
        outcome = 1
        hit_barrier = 'tp'
        horizon_bars = tp_bar
        exit_price = tp_price
        return_pct = (tp_price - entry_price) / entry_price
    elif sl_bar <= max_horizon_bars:
        # Stop loss hit first
        outcome = -1
        hit_barrier = 'sl'
        horizon_bars = sl_bar
        exit_price = sl_price
        return_pct = (sl_price - entry_price) / entry_price
    else:
        # Timeout
        outcome = 0
        hit_barrier = 'timeout'
        horizon_bars = max_horizon_bars
        exit_price = future_prices.iloc[-1]
        return_pct = (exit_price - entry_price) / entry_price
    
    return TripleBarrierResult(
        outcome=outcome,
        hit_barrier=hit_barrier,
        horizon_bars=horizon_bars,
        exit_price=exit_price,
        return_pct=return_pct
    )


def label_trades_batch(
    prices: pd.Series,
    entry_indices: List[int],
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_horizon_bars: int = 20
) -> List[TripleBarrierResult]:
    """
    Label multiple trades using the triple-barrier method.
    
    Args:
        prices: Price series
        entry_indices: List of entry bar indices
        pt_mult: Take profit multiplier
        sl_mult: Stop loss multiplier
        max_horizon_bars: Maximum bars to wait
        
    Returns:
        List of TripleBarrierResult objects
    """
    results = []
    for entry_idx in entry_indices:
        result = label_trades(
            prices=prices,
            entry_idx=entry_idx,
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            max_horizon_bars=max_horizon_bars
        )
        results.append(result)
    return results


def calculate_outcome_statistics(
    results: List[TripleBarrierResult]
) -> Dict[str, float]:
    """
    Calculate statistics from triple-barrier results.
    
    Args:
        results: List of TripleBarrierResult objects
        
    Returns:
        Dictionary with win_rate, avg_return, etc.
    """
    if not results:
        return {
            'win_rate': 0.0,
            'avg_return': 0.0,
            'avg_horizon': 0.0,
            'tp_rate': 0.0,
            'sl_rate': 0.0,
            'timeout_rate': 0.0
        }
    
    outcomes = [r.outcome for r in results]
    returns = [r.return_pct for r in results]
    horizons = [r.horizon_bars for r in results]
    barriers = [r.hit_barrier for r in results]
    
    win_rate = np.mean([o == 1 for o in outcomes])
    avg_return = np.mean(returns)
    avg_horizon = np.mean(horizons)
    tp_rate = np.mean([b == 'tp' for b in barriers])
    sl_rate = np.mean([b == 'sl' for b in barriers])
    timeout_rate = np.mean([b == 'timeout' for b in barriers])
    
    return {
        'win_rate': win_rate,
        'avg_return': avg_return,
        'avg_horizon': avg_horizon,
        'tp_rate': tp_rate,
        'sl_rate': sl_rate,
        'timeout_rate': timeout_rate
    }


def validate_triple_barrier(
    prices: pd.Series,
    entry_idx: int,
    expected_outcome: int,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_horizon_bars: int = 20
) -> bool:
    """
    Validate triple-barrier labeling against expected outcome.
    
    Args:
        prices: Price series
        entry_idx: Entry bar index
        expected_outcome: Expected outcome (+1, -1, 0)
        pt_mult: Take profit multiplier
        sl_mult: Stop loss multiplier
        max_horizon_bars: Maximum horizon
        
    Returns:
        True if actual outcome matches expected
    """
    result = label_trades(
        prices=prices,
        entry_idx=entry_idx,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        max_horizon_bars=max_horizon_bars
    )
    return result.outcome == expected_outcome
