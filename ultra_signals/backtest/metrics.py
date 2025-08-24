from __future__ import annotations

import math
from typing import List, Dict, Any, Optional, Iterable

import numpy as np
import pandas as pd

# Keep for compatibility (unused by tests but imported elsewhere)
from ultra_signals.core.custom_types import TradeRecord, EquityDataPoint, ReliabilityReport  # noqa: F401


def compute_kpis(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes key performance indicators from a DataFrame of trades.

    Tests expect:
      - if empty: dict containing key "error"
      - keys: total_pnl, total_trades, win_rate_pct, profit_factor, average_win, average_loss
    """
    if trades_df is None or trades_df.empty:
        return {
            "error": "no trades",
            "total_pnl": 0.0,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "avg_pnl": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "avg_hold_bars": 0.0,
        }

    df = trades_df.copy()

    # normalize column names
    if "pnl" not in df.columns:
        if "PnL" in df.columns:
            df["pnl"] = df["PnL"]
        else:
            df["pnl"] = 0.0
    if "bars_held" not in df.columns:
        if "hold_bars" in df.columns:
            df["bars_held"] = df["hold_bars"]
        else:
            df["bars_held"] = 0

    total_trades = int(len(df))
    total_pnl = float(df["pnl"].sum())

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    n_wins = int(len(wins))
    n_losses = int(len(losses))

    win_rate_pct = (n_wins / total_trades) * 100.0 if total_trades > 0 else 0.0
    avg_pnl = float(df["pnl"].mean()) if total_trades > 0 else 0.0
    average_win = float(wins["pnl"].mean()) if n_wins > 0 else 0.0
    average_loss = float(losses["pnl"].mean()) if n_losses > 0 else 0.0
    avg_hold_bars = float(df["bars_held"].mean()) if total_trades > 0 else 0.0

    gross_win = float(wins["pnl"].sum()) if n_wins > 0 else 0.0
    gross_loss_abs = float(abs(losses["pnl"].sum())) if n_losses > 0 else 0.0
    if gross_loss_abs > 0:
        profit_factor = gross_win / gross_loss_abs
    else:
        profit_factor = float("inf") if gross_win > 0 else 0.0

    # simple Sharpe proxy using pnl per trade
    ret = df["pnl"].astype(float)
    if ret.std(ddof=0) > 0:
        sharpe = float((ret.mean() / ret.std(ddof=0)) * math.sqrt(252))
    else:
        sharpe = 0.0

    # max drawdown on cumulative pnl
    cum = ret.cumsum()
    peak = cum.cummax()
    drawdown = cum - peak
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    return {
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "win_rate_pct": win_rate_pct,
        "avg_pnl": avg_pnl,
        "average_win": average_win,
        "average_loss": average_loss,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_hold_bars": avg_hold_bars,
    }


def generate_equity_curve(first_arg, initial_capital: Optional[float] = None) -> pd.Series:
    """
    Polymorphic helper returning a **pd.Series** (as tests expect).

    Two supported usages:
      1) generate_equity_curve(trades_df, initial_capital)
         - trades_df: has columns ['exit_time', 'pnl']
         - returns Series indexed by 'exit_time' (sorted) of equity values

      2) generate_equity_curve(equity_data_list)
         - equity_data_list: list of dicts with 'timestamp' and 'equity'
         - returns Series indexed by 'timestamp'
    """
    # Case 2: equity_data list (used by CLI reporting)
    if isinstance(first_arg, (list, tuple)):
        equity_data: Iterable[Dict[str, Any]] = first_arg
        if not equity_data:
            return pd.Series([], dtype=float)
        df = pd.DataFrame(equity_data)
        if not {"timestamp", "equity"}.issubset(df.columns):
            raise ValueError("equity_data must contain 'timestamp' and 'equity'.")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp")
        return df["equity"].astype(float)

    # Case 1: trades df + initial capital (used by KPI tests)
    trades = first_arg
    if not isinstance(trades, pd.DataFrame):
        raise TypeError("First argument must be a trades DataFrame or a list of equity data dicts.")
    if initial_capital is None:
        raise TypeError("generate_equity_curve(trades_df, initial_capital) requires initial_capital.")

    if trades.empty:
        return pd.Series([], dtype=float)

    df = trades.copy()
    if "exit_time" not in df.columns:
        raise ValueError("trades must have 'exit_time' column")
    if "pnl" not in df.columns:
        if "PnL" in df.columns:
            df["pnl"] = df["PnL"]
        else:
            raise ValueError("trades must have 'pnl' column")

    df = df.sort_values("exit_time").reset_index(drop=True)
    equity = float(initial_capital) + df["pnl"].astype(float).cumsum()
    s = pd.Series(equity.values, index=pd.to_datetime(df["exit_time"]))
    s.index.name = "timestamp"
    s.name = "equity"
    return s


def calculate_brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def compute_reliability_bins(predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(predictions, bins[1:-1])

    bin_sums = np.bincount(bin_ids, weights=predictions, minlength=n_bins)
    bin_true = np.bincount(bin_ids, weights=outcomes, minlength=n_bins)
    bin_counts = np.bincount(bin_ids, minlength=n_bins)

    non_empty = bin_counts > 0
    mean_predicted_value = np.full(n_bins, np.nan, dtype=float)
    fraction_of_positives = np.full(n_bins, np.nan, dtype=float)

    mean_predicted_value[non_empty] = bin_sums[non_empty] / bin_counts[non_empty]
    fraction_of_positives[non_empty] = bin_true[non_empty] / bin_counts[non_empty]

    brier = calculate_brier_score(outcomes, predictions)

    return {
        "bins": {
            "mean_predicted": mean_predicted_value,
            "fraction_positives": fraction_of_positives,
            "counts": bin_counts.astype(int),
        },
        "brier_score": brier,
    }
