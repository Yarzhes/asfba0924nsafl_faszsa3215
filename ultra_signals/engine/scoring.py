"""
Scoring Engine for Signal Generation
- Works with FeatureStore dataclasses (backtest) and plain dicts (tests).
"""
from typing import Dict, Optional
import math
import numpy as np

from ultra_signals.core.custom_types import FeatureVector, DerivativesFeatures, RegimeFeatures
from ultra_signals.features.orderbook import OrderbookFeatures
from loguru import logger


# ---------- helpers ------------------------------------------------------------

def _sig(x: float, a: float = 2.0) -> float:
    """Simple sigmoid-like function to squash a value to [-1, 1]."""
    return (2 / (1 + math.exp(-a * x))) - 1

def _get(src, attr: str = None, key: str = None, default=None):
    """
    Try to read `attr` from an object (dataclass) or `key` from a dict.
    Returns `default` if nothing is found.
    """
    if src is None:
        return default
    if attr is not None and hasattr(src, attr):
        return getattr(src, attr)
    if isinstance(src, dict) and key is not None:
        return src.get(key, default)
    return default


# ---------- component scores ---------------------------------------------------

def trend_score(src, params: Dict) -> float:
    """
    EMA alignment score.
    - Supports dataclass inputs with attributes: ema_short, ema_medium, ema_long
    - Also supports dict inputs with keys: ema_{period}
    """
    p = params.get("trend", {})
    ema_s = _get(src, "ema_short", f"ema_{p.get('ema_short', 10)}")
    ema_m = _get(src, "ema_medium", f"ema_{p.get('ema_medium', 20)}")
    ema_l = _get(src, "ema_long", f"ema_{p.get('ema_long', 50)}")
    if None in (ema_s, ema_m, ema_l):
        return 0.0

    # Perfect alignment → ±1; otherwise partial credit
    if ema_s > ema_m > ema_l:
        return 1.0
    if ema_s < ema_m < ema_l:
        return -1.0

    score = 0.0
    score += 0.5 if ema_s > ema_m else -0.5
    score += 0.5 if ema_m > ema_l else -0.5
    return float(np.clip(score, -1.0, 1.0))


def momentum_score(src, params: Dict) -> float:
    """
    Momentum from RSI & MACD histogram.
    - Dataclass: uses attributes rsi, macd_hist
    - Dict: uses rsi_{period}, macd_hist
    - RSI drives the sign; MACD is confirmation (light weight).
    """
    p = params.get("momentum", {})
    rsi = float(_get(src, "rsi", f"rsi_{p.get('rsi_period', 14)}", 50.0))
    macd_hist = float(_get(src, "macd_hist", "macd_hist", 0.0))

    rsi_part = (rsi - 50.0) / 50.0                  # ~[-1, 1]
    macd_part = float(np.clip(macd_hist * 10.0, -1.0, 1.0))
    score = 0.8 * rsi_part + 0.2 * macd_part

    # If RSI is clearly bearish but MACD slightly bullish, keep a small negative tilt.
    if rsi < 45 and macd_hist < 0.2 and score > 0:
        score = min(score, -0.05)

    return float(np.clip(score, -1.0, 1.0))


def volatility_score(_src, _params: Dict) -> float:
    """Placeholder for volatility-related scoring."""
    return 0.0


def orderbook_score(features: Optional[OrderbookFeatures]) -> float:
    """Score from order book imbalance."""
    if not features or getattr(features, "imbalance", None) is None:
        return 0.0
    return _sig(features.imbalance - 1, a=2.0)


def _funding_contrarian_score(funding_rate: Optional[float]) -> float:
    if funding_rate is None:
        return 0.0
    return -np.sign(funding_rate) * 0.1


def _oi_divergence_score(oi_delta: Optional[float], price_change: float) -> float:
    if oi_delta is None:
        return 0.0
    return np.sign(oi_delta * price_change) * 0.1


def _liq_pulse_score(liq_pulse: int) -> float:
    return liq_pulse * 0.3


def derivatives_score(features: Optional[DerivativesFeatures], ohlcv: Dict) -> float:
    """Composite score from derivatives features."""
    if not features:
        return 0.0
    price_change = ohlcv.get("close", 0) - ohlcv.get("open", 0)
    funding_score = _funding_contrarian_score(getattr(features, "funding_now", None))
    oi_score = _oi_divergence_score(getattr(features, "oi_delta_5m", None), price_change)
    liq_score = _liq_pulse_score(getattr(features, "liq_pulse", 0))
    return float(np.clip(np.mean([funding_score, oi_score, liq_score]), -1.0, 1.0))


def trend_pullback_score(ohlcv_features: dict, orderbook_features: Optional[OrderbookFeatures], params: Dict) -> float:
    """Scores a trend-pullback scenario based on VWAP and order book confluence."""
    if not orderbook_features:
        return 0.0
    vwap = ohlcv_features.get("vwap_session", np.nan)
    upper_band = ohlcv_features.get("vwap_session_upper1", np.nan)
    lower_band = ohlcv_features.get("vwap_session_lower1", np.nan)
    close = ohlcv_features.get("close", np.nan)
    if any(np.isnan(v) for v in [vwap, upper_band, lower_band, close]):
        return 0.0

    score = 0.0
    ema_long_key = f"ema_{params.get('trend', {}).get('ema_long', 50)}"
    ema_long_val = ohlcv_features.get(ema_long_key, 0)

    if vwap > ema_long_val and close <= lower_band:
        if getattr(orderbook_features, "imbalance", None) and orderbook_features.imbalance > 0.6:
            score = 1.0
    elif vwap < ema_long_val and close >= upper_band:
        if getattr(orderbook_features, "imbalance", None) and orderbook_features.imbalance < 0.4:
            score = -1.0
    return score


def breakout_score(cvd_features: Optional[dict], orderbook_features: Optional[OrderbookFeatures]) -> float:
    """Scores a breakout scenario based on book-flip and CVD slope."""
    if not cvd_features or not orderbook_features:
        return 0.0
    cvd_slope = cvd_features.get("cvd_slope", 0.0)
    book_flip = getattr(orderbook_features, "book_flip", 0)
    if book_flip == 1 and cvd_slope > 0:
        return 1.0
    if book_flip == -1 and cvd_slope < 0:
        return -1.0
    return 0.0


def relative_strength_gate(base_score: float, rs_features: dict, symbol: str) -> float:
    """Applies a boost or penalty based on the asset's relative strength ranking."""
    if not rs_features:
        return base_score
    if base_score > 0:
        return base_score * (1.5 if symbol in rs_features.get("top_k_longs", []) else 0.5)
    if base_score < 0:
        return base_score * (1.5 if symbol in rs_features.get("bottom_k_shorts", []) else 0.5)
    return base_score


def component_scores(features: FeatureVector, config_params: Dict) -> Dict[str, float]:
    """
    Use dataclasses when available (backtest); fall back to `features.ohlcv` (tests).
    """
    ohlcv_dict = getattr(features, "ohlcv", {}) or {}

    # prefer FeatureStore components; else the test dict
    trend_src = getattr(features, "trend", None) or ohlcv_dict
    mom_src   = getattr(features, "momentum", None) or ohlcv_dict
    vol_src   = getattr(features, "volatility", None) or {}

    trend = trend_score(trend_src, config_params)
    momentum = momentum_score(mom_src, config_params)
    volatility = volatility_score(vol_src, config_params)
    orderbook = orderbook_score(getattr(features, "orderbook", None))
    derivatives = derivatives_score(getattr(features, "derivatives", None), ohlcv_dict)

    pullback = trend_pullback_score(getattr(features, "volume_flow", {}) or {}, getattr(features, "orderbook", None), config_params)
    breakout = breakout_score(getattr(features, "derivatives", None), getattr(features, "orderbook", None))

    scores = {
        "trend": trend,
        "momentum": momentum,
        "volatility": volatility,
        "orderbook": orderbook,
        "derivatives": derivatives,
        "pullback_confluence": pullback,
        "breakout_confluence": breakout,
    }

    rs = getattr(features, "rs", {}) or {}
    symbol = getattr(features, "symbol", "")
    scores["pullback_confluence_rs"] = relative_strength_gate(pullback, rs, symbol)

    logger.debug(f"Component scores for {symbol}: {scores}")
    return scores
