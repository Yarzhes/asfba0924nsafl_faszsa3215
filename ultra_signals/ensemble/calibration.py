"""Calibration helpers: Platt scaling, isotonic wrapper, ECE/Brier metrics and reliability plot."""
from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt


def platt_scaler(probs: np.ndarray, y: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(probs.reshape(-1, 1), y)
    return lr


def isotonic_scaler(probs: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(probs, y)
    return iso


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    return np.mean((probs - y) ** 2)


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(probs, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if not mask.any():
            continue
        bin_prob = probs[mask].mean()
        bin_true = y[mask].mean()
        ece += (mask.mean()) * abs(bin_prob - bin_true)
    return float(ece)


def reliability_plot(probs: np.ndarray, y: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(probs, bins) - 1
    xs = []
    ys = []
    counts = []
    for i in range(n_bins):
        mask = binids == i
        if not mask.any():
            xs.append((bins[i] + bins[i + 1]) / 2.0)
            ys.append(np.nan)
            counts.append(0)
            continue
        xs.append((bins[i] + bins[i + 1]) / 2.0)
        ys.append(y[mask].mean())
        counts.append(mask.sum())
    plt.figure()
    plt.plot(xs, ys, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed frequency')
    plt.title('Reliability plot')
    return plt.gcf()


def save_reliability_plot(probs: np.ndarray, y: np.ndarray, path: str, n_bins: int = 10):
    """Create and save a reliability plot to `path` and return the path.

    This helper ensures the figure is closed after saving to avoid open GUI
    resources during test runs or CI.
    """
    fig = reliability_plot(probs, y, n_bins=n_bins)
    fig.savefig(path)
    # close to free resources
    try:
        import matplotlib.pyplot as _plt

        _plt.close(fig)
    except Exception:
        pass
    return path
