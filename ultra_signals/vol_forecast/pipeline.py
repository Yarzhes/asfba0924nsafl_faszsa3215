from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .models import VolModelManager
from .realized import parkinson, garman_klass, rogers_satchell


def annualize_sigma(sigma_per_bar: float, bars_per_year: int = 525600) -> float:
    """Convert per-bar sigma to annualized sigma (simple sqrt scaling)."""
    return float(sigma_per_bar) * np.sqrt(bars_per_year)


def confidence_interval(sigma: float, n: int = 30, ci: float = 0.95) -> Tuple[float, float]:
    """Approximate confidence interval for sigma using chi-square approx on variance.

    This is a quick approximation: var ~ sigma^2; use chi2 to get CI for variance then sqrt.
    """
    from scipy.stats import chi2

    alpha = 1 - ci
    df = max(1, n - 1)
    lower_var = df * (sigma ** 2) / chi2.ppf(1 - alpha / 2, df)
    upper_var = df * (sigma ** 2) / chi2.ppf(alpha / 2, df)
    return float(np.sqrt(lower_var)), float(np.sqrt(upper_var))


def walk_forward_refit_and_score(df: pd.DataFrame, model_type: str = "garch", train_window: int = 500, test_window: int = 100, embargo: int = 10, purge: int = 5) -> Dict[str, Any]:
    """Perform a simple walk-forward refit and compute OOS RMSE and pinball loss on sigma forecasts.

    Returns dict with oos_score (rmse) and metadata.
    """
    prices = df["close"].astype(float)
    returns = prepare_returns(df)

    n = len(returns)
    if n < train_window + test_window:
        return {"oos_rmse": None, "oos_pinball": None}

    rmse_list = []
    pinball_list = []
    from math import isfinite

    for start in range(0, n - train_window - test_window + 1, test_window):
        train_idx = slice(start, start + train_window)
        test_idx = slice(start + train_window + embargo, start + train_window + embargo + test_window)

        r_train = returns.iloc[train_idx]
        r_test = returns.iloc[test_idx]
        if r_test.empty or r_train.empty:
            continue

        mgr = VolModelManager(model_type)
        try:
            mgr.fit(r_train)
            # forecast per-bar sigma for horizon=1 for each test row (naive repeating forecast)
            sigma_hat = mgr.forecast_sigma(1)
            # realized sigma in test window: rolling std per-bar
            realized_sigma = r_test.rolling(30).std().dropna()
            if realized_sigma.empty:
                continue
            # align lengths
            s_hat = np.repeat(sigma_hat, len(realized_sigma))
            rmse = float(np.sqrt(np.mean((s_hat - realized_sigma.values) ** 2)))
            rmse_list.append(rmse)

            # pinball loss for VaR (use median, tau=0.95 -> quantile). Simple symmetric approx using normal.
            tau = 0.95
            q = np.quantile(r_test.values, tau)
            residuals = r_test.values - q
            pin = np.mean(np.maximum(tau * residuals, (tau - 1) * residuals))
            pinball_list.append(float(pin))
        except Exception:
            continue

    oos_rmse = float(np.mean(rmse_list)) if rmse_list else None
    oos_pinball = float(np.mean(pinball_list)) if pinball_list else None
    return {"oos_rmse": oos_rmse, "oos_pinball": oos_pinball}


def prepare_returns(df: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Compute log-returns with robust de-meaning and optional microstructure filter.

    Expects DataFrame with datetime index and a column named `price_col`.
    Returns float returns (not percentage).
    """
    p = df[price_col].astype(float)
    r = np.log(p).diff()
    r = r - r.median()
    # optional tiny-return filter
    r[np.abs(r) < 1e-8] = 0.0
    return r.dropna()


def forecast_vols(df: pd.DataFrame, model_types: List[str] = None, horizons: List[int] = None) -> Dict[str, Any]:
    """Fit models and return sigma forecasts and metadata.

    Returns a dict containing per-model sigma forecasts and a chosen model.
    """
    if model_types is None:
        model_types = ["egarch", "garch", "tgarch"]
    if horizons is None:
        horizons = [1]

    returns = prepare_returns(df)

    results = {}
    for mt in model_types:
        mgr = VolModelManager(mt)
        meta = mgr.fit(returns)
        sigs = {f"{mt}_sigma_{h}": mgr.forecast_sigma(h) for h in horizons}
        # compute OOS score using quick walk-forward
        score = walk_forward_refit_and_score(df, mt)
        results[mt] = {"meta": meta, "sigmas": sigs, "score": score}

    # choose best available model: prefer EGARCH -> GARCH -> realized (as per spec)
    for preferred in ["egarch", "garch", "tgarch"]:
        if preferred in results:
            chosen = preferred
            break
    else:
        chosen = list(results.keys())[0]

    out = {
        "model_choice": chosen,
        "forecasts": results[chosen]["sigmas"],
        "models": results,
    }

    # VaR (normal approx) for next bar using sigma_1
    sigma_next = out["forecasts"].get(f"{chosen}_sigma_1")
    if sigma_next is not None:
        z95 = 1.645
        z99 = 2.33
        out["VaR95_next"] = z95 * sigma_next
        out["VaR99_next"] = z99 * sigma_next
        # annualized and confidence
        out["sigma_annual"] = annualize_sigma(sigma_next)
        try:
            ci_low, ci_high = confidence_interval(sigma_next, n=30, ci=0.95)
            out["sigma_conf"] = {"low": ci_low, "high": ci_high}
        except Exception:
            out["sigma_conf"] = None

    # regime tag (z-score of sigma vs recent mean)
    # compute a quick realized sigma window for tagging
    realized = returns.rolling(30).std().dropna()
    if not realized.empty:
        mu = float(realized.iloc[-1])
        sigma_z = (sigma_next - mu) / (np.std(realized) + 1e-12) if sigma_next is not None else 0.0
        if sigma_z < -0.5:
            regime = "low"
        elif sigma_z > 0.5:
            regime = "high"
        else:
            regime = "med"
        out["vol_regime"] = regime
        out["sigma_z"] = float(sigma_z)

    # fallback: if chosen model has no good OOS score, use realized estimators
    chosen_score = results[chosen].get("score", {})
    if chosen_score and chosen_score.get("oos_rmse") is not None:
        if chosen_score["oos_rmse"] > np.nanpercentile([v.get("score", {}).get("oos_rmse") or np.inf for v in results.values()], 90):
            # poor performing model -> fallback to Parkinson realized vol
            try:
                pv = parkinson(df).iloc[-1]
                out["forecasts"] = {f"realized_parkinson_sigma_1": float(pv)}
                out["model_choice"] = "realized_parkinson"
            except Exception:
                pass

    return out
