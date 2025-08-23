import pandas as pd
import numpy as np
from typing import List, Dict, Any
from ultra_signals.core.custom_types import TradeRecord, EquityDataPoint, ReliabilityReport

def compute_kpis(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes key performance indicators from a DataFrame of trades."""
    if trades_df.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "avg_hold_bars": 0.0,
        }

    total_trades = len(trades_df)
    total_pnl = trades_df["pnl"].sum()
    
    wins = trades_df[trades_df["pnl"] > 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    
    avg_pnl = trades_df["pnl"].mean()
    avg_hold_bars = trades_df["hold_bars"].mean() if "hold_bars" in trades_df.columns else 0.0

    # Max Drawdown
    equity_curve = trades_df["pnl"].cumsum()
    max_equity = equity_curve.expanding().max()
    drawdown = max_equity - equity_curve
    max_drawdown = drawdown.max()

    # Sharpe Ratio (simplified, dailyized proxy)
    # Assuming pnl is per-trade, not daily. For a true Sharpe, need daily returns.
    # Here, we'll use trade-level pnl std dev.
    sharpe = trades_df["pnl"].mean() / trades_df["pnl"].std() if trades_df["pnl"].std() != 0 else 0.0
    # Annualize if trades represent daily periods (rough approximation)
    # sharpe *= np.sqrt(252)

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "avg_hold_bars": avg_hold_bars,
    }

def generate_equity_curve(equity_data: List[Dict]) -> pd.Series:
    """Generates an equity curve from a list of equity data points."""
    if not equity_data:
        return pd.Series([], dtype=float)
    
    df = pd.DataFrame(equity_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df["equity"]

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