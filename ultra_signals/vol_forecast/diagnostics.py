import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any


def plot_predicted_vs_realized(dates, predicted, realized, title: str = "Predicted vs Realized Sigma"):
    plt.figure(figsize=(10, 4))
    plt.plot(dates, predicted, label="predicted")
    plt.plot(dates, realized, label="realized", alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("sigma")
    plt.tight_layout()
    return plt.gcf()


def var_breach_rate(returns: pd.Series, var_series: pd.Series) -> float:
    """Compute fraction of returns that breached VaR (i.e., return < -VaR)."""
    breaches = (returns < -var_series).sum()
    total = len(returns.dropna())
    return float(breaches) / total if total else 0.0
