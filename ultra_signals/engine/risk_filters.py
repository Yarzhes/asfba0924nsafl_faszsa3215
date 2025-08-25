"""
Risk Management Filters

Provides the apply_filters function to check if a signal passes basic risk filters.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from ultra_signals.engine.confluence import confluence_htf_agrees
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import Signal

# ----------------- ORIGINAL DATACLASS (kept) -----------------
@dataclass
class FilterResult:
    passed: bool
    reason: str = ""
    details: Optional[Dict] = None


# ----------------- SPRINT 8: SAFE HELPERS (non-breaking) -----------------
def _safe_store_call(store: FeatureStore, method_name: str, *args, **kwargs):
    """
    Call a FeatureStore method if it exists; otherwise return None.
    Prevents crashes if a metric isn't implemented yet.
    """
    fn = getattr(store, method_name, None)
    if callable(fn):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None
    return None


def _htf_confluence_agrees(signal: Signal, store: FeatureStore, settings: dict) -> bool:
    """
    Require higher timeframe (HTF) regime to agree with signal direction, if configured.
    - Uses settings.confluence.map (e.g. {'15m':'1h', '1h':'4h'})
    - Expects FeatureStore to provide get_regime(symbol, timeframe) if available.
      If not available or unknown, we allow (return True) to avoid blocking everything.
    """
    c_cfg = settings.get("confluence", {}) or {}
    tf_map = c_cfg.get("map", {}) or {}
    require_align = bool(c_cfg.get("require_regime_align", True))
    if not require_align:
        return True

    htf = tf_map.get(signal.timeframe)
    if not htf:
        return True  # no mapping -> don't block

    regime = _safe_store_call(store, "get_regime", signal.symbol, htf)
    # Common regimes might be: 'trend_up', 'trend_down', 'mr', 'mixed', etc.
    if regime is None:
        return True  # unknown -> don't block

    # Simple alignment policy: trend_up blocks SHORT; trend_down blocks LONG; otherwise allow
    if regime == "trend_up" and signal.decision == "SHORT":
        return False
    if regime == "trend_down" and signal.decision == "LONG":
        return False
    return True


def apply_filters(signal: Signal, store: FeatureStore, settings: dict) -> FilterResult:
    """
    Checks if a signal passes basic risk filters.
    - Must check: warmup bars and spread (bid/ask) availability.
    - `store` is FeatureStore, use its stored OHLCV and book ticker.
    - `settings` is a dict (pydantic model dumped by .model_dump()).
    """
    # ----------------- 1. WARMUP CHECK -----------------
    warmup_periods = settings.get("features", {}).get("warmup_periods", 20)
    available_bars = store.get_warmup_status(signal.symbol, signal.timeframe)
    if available_bars < warmup_periods:
        return FilterResult(False, reason="WARMUP_INCOMPLETE")

    # ----------------- 2. BOOK TICKER VALIDATION -----------------
    book_ticker = store.get_book_ticker(signal.symbol)
    if not book_ticker:
        return FilterResult(False, reason="MISSING_BOOK_TICKER")

    # Ensure book_ticker is a tuple with at least 2 elements
    if not isinstance(book_ticker, tuple) or len(book_ticker) < 2:
        return FilterResult(False, reason="INVALID_BOOK_TICKER")

    bid, ask = book_ticker[:2]
    if bid <= 0 or ask <= 0:
        return FilterResult(False, reason="INVALID_PRICE")

    mid_price = (ask + bid) / 2
    if mid_price == 0:
        return FilterResult(False, reason="ZERO_MID_PRICE")

    # ----------------- 3. SPREAD CALCULATION -----------------
    spread = ask - bid
    spread_pct = spread / mid_price

    # ----------------- 4. MAX SPREAD FETCH (UPDATED) -----------------
    # Try to fetch max_spread_pct from top-level filters first.
    # If not found, fallback to engine → risk → max_spread_pct → default.
    max_spread_pct = settings.get("filters", {}).get("max_spread_pct")
    if max_spread_pct is None:
        max_spread_pct = (
            settings.get("engine", {})
                    .get("risk", {})
                    .get("max_spread_pct", {})
                    .get("default", 0.002)  # fallback hard default = 0.2%
        )

    # ----------------- 5. APPLY SPREAD % FILTER -----------------
    if spread_pct > max_spread_pct:
        return FilterResult(
            False,
            reason="SPREAD_TOO_WIDE",
            details={"spread": spread_pct, "max_allowed": max_spread_pct}
        )

    # =================================================================
    # ============= SPRINT 8 ADDITIONS (OPTIONAL, NON-BREAKING) ========
    # =================================================================
    # We'll collect additional gating reasons. If any are present, we block.
    reasons: List[str] = []
    details: Dict[str, Any] = {
        "spread_pct": spread_pct
    }

    # --- 6.1 ATR Percentile Gate ---
    # Only trade if ATR percentile is high enough (volatility present).
    atr_gate_pct = int(settings.get("filters", {}).get("atr_gate_pct", 0))  # default 0 = disabled
    atr_pct = _safe_store_call(store, "get_atr_percentile", signal.symbol, signal.timeframe)
    details["atr_percentile"] = atr_pct
    if atr_gate_pct > 0 and atr_pct is not None and atr_pct < atr_gate_pct:
        reasons.append("LOW_ATR")

    # --- 6.2 ADX Trend Strength Gate ---
    adx_min = int(settings.get("filters", {}).get("adx_min", 0))  # default 0 = disabled
    adx_val = _safe_store_call(store, "get_adx", signal.symbol, signal.timeframe)
    details["adx"] = adx_val
    if adx_min > 0 and adx_val is not None and adx_val < adx_min:
        reasons.append("LOW_ADX")

    # --- 6.3 TR Compression (skip ultra-chop) ---
    tr_comp_max = settings.get("filters", {}).get("tr_compression_max", None)
    tr_comp = _safe_store_call(store, "get_tr_compression", signal.symbol, signal.timeframe)
    details["tr_compression"] = tr_comp
    if tr_comp_max is not None and tr_comp is not None:
        try:
            tr_comp_max_f = float(tr_comp_max)
            if tr_comp <= tr_comp_max_f:
                reasons.append("TR_COMPRESSION")
        except Exception:
            pass  # ignore malformed config

    # --- 6.4 Funding Window Avoidance ---
    # Avoid opening too close to the next funding event (minutes window).
    window_min = int(settings.get("veto", {}).get("near_funding_window_min", 0))  # 0 = disabled
    mins_to_funding = _safe_store_call(store, "get_minutes_to_next_funding", signal.symbol)
    details["mins_to_funding"] = mins_to_funding
    if window_min > 0 and mins_to_funding is not None and abs(mins_to_funding) < window_min:
        reasons.append("NEAR_FUNDING_WINDOW")

    # --- 6.5 Wide Spread (bps) Veto (separate from % check above) ---
    wide_spread_bps = int(settings.get("veto", {}).get("wide_spread_bps", 0))  # 0 = disabled
    # If your store can provide spread in basis points:
    spread_bps = _safe_store_call(store, "get_spread_bps", signal.symbol)
    details["spread_bps"] = spread_bps
    if wide_spread_bps > 0 and spread_bps is not None and spread_bps > wide_spread_bps:
        reasons.append("WIDE_SPREAD")

    # --- 6.6 Multi-Timeframe Confluence ---
    # Require HTF regime alignment if enabled.
    confluence_required = bool(settings.get("confluence", {}).get("require_regime_align", True))
    if confluence_required:
        if not _htf_confluence_agrees(signal, store, settings):
            reasons.append("MTF_DISAGREE")

    # If any of the Sprint-8 reasons were triggered, veto the trade:
    if reasons:
        return FilterResult(False, reason=";".join(reasons), details=details)

    # ----------------- 7. PASSED -----------------
    return FilterResult(True)
