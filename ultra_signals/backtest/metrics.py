import pandas as pd
import numpy as np
from typing import List, Dict, Any
from ultra_signals.core.custom_types import TradeRecord, EquityDataPoint, ReliabilityReport

def compute_kpis(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes key performance indicators from a DataFrame of trades."""
    if trades_df.empty:
        return {"error": "No trades to analyze."}

    pnl = trades_df['pnl']
    total_pnl = pnl.sum()
    total_trades = len(trades_df)
    
    wins = trades_df[pnl > 0]
    losses = trades_df[pnl <= 0]
    
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
    
    profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() != 0 else np.inf
    
    # Simplified Sharpe Ratio (assuming risk-free rate is 0)
    sharpe_ratio = pnl.mean() / pnl.std() if pnl.std() != 0 else 0
    
    return {
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "win_rate_pct": win_rate * 100,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio * np.sqrt(252), # Annualized
        "average_win": avg_win,
        "average_loss": avg_loss,
    }

def generate_equity_curve(trades_df: pd.DataFrame, initial_capital: float = 10000.0) -> pd.Series:
    """Generates an equity curve from trades."""
    if trades_df.empty:
        return pd.Series([initial_capital], index=[pd.Timestamp.now()])
        
    pnl_curve = trades_df['pnl'].cumsum()
    equity_curve = initial_capital + pnl_curve
    equity_curve.index = trades_df['exit_time']
    return equity_curve

def calculate_brier_score(y_true, y_prob):
    """Calculates the Brier score for reliability."""
    return np.mean((y_prob - y_true) ** 2)

def compute_reliability_bins(predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Computes reliability bins and Brier score."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(predictions, bins[1:-1])

    bin_sums = np.bincount(bin_ids, weights=predictions, minlength=n_bins)
    bin_true = np.bincount(bin_ids, weights=outcomes, minlength=n_bins)
    bin_counts = np.bincount(bin_ids, minlength=n_bins)

    # Avoid division by zero
    non_empty_bins = bin_counts > 0
    mean_predicted_value = np.full(n_bins, np.nan)
    fraction_of_positives = np.full(n_bins, np.nan)
    
    mean_predicted_value[non_empty_bins] = bin_sums[non_empty_bins] / bin_counts[non_empty_bins]
    fraction_of_positives[non_empty_bins] = bin_true[non_empty_bins] / bin_counts[non_empty_bins]

    brier = calculate_brier_score(outcomes, predictions)

    return {
        "bins": {
            "mean_predicted": mean_predicted_value,
            "fraction_positives": fraction_of_positives,
            "counts": bin_counts
        },
        "brier_score": brier
    }