""""
Risk Management Filters

Provides the apply_filters function to check if a signal passes basic risk filters.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from ultra_signals.engine.confluence import confluence_htf_agrees
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import Signal

# ========= SPRINT 9: SAFE IMPORTS (won't crash if modules missing) =========
# We import new helpers defensively so your file doesn't break if a module
# hasn't been created yet. If an import fails, we set it to None and
# simply skip that specific veto at runtime.
try:
    from ultra_signals.data.funding_provider import FundingProvider  # minutes_to_next()
except Exception:
    FundingProvider = None  # type: ignore

try:
    from ultra_signals.orderflow.cvd import CVDComputer  # compute_proxy()
except Exception:
    CVDComputer = None  # type: ignore

try:
    from ultra_signals.liquidity.liquidations import LiqPulse  # latest_spike_flag()
except Exception:
    LiqPulse = None  # type: ignore

try:
    from ultra_signals.marketdata.depth_agg import DepthAggregator  # evaluate()
except Exception:
    DepthAggregator = None  # type: ignore

try:
    from ultra_signals.analytics.news_filter import NewsFilter  # is_blocked()
except Exception:
    NewsFilter = None  # type: ignore


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

    # --- 6.4 Funding Window Avoidance (Sprint 8 baseline) ---
    # Prefer store-provided minutes if available; otherwise (Sprint 9) fall back to FundingProvider.
    window_min = int(settings.get("veto", {}).get("near_funding_window_min", 0))  # 0 = disabled
    mins_to_funding = _safe_store_call(store, "get_minutes_to_next_funding", signal.symbol)

    # SPRINT 9: if store doesn't provide it, query FundingProvider directly
    if mins_to_funding is None and FundingProvider is not None:
        try:
            fp = FundingProvider(settings)
            # Try to get a notion of "now" from store; otherwise None
            now_ms = (
                _safe_store_call(store, "current_ts_ms", signal.symbol, signal.timeframe)
                or _safe_store_call(store, "get_current_ts_ms", signal.symbol, signal.timeframe)
            )
            if now_ms is not None:
                mins_to_funding = fp.minutes_to_next(signal.symbol, now_ms)
        except Exception:
            mins_to_funding = None

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

    # =================================================================
    # ======================= SPRINT 9 VETO BLOCKS =====================
    # =================================================================

    # Helper to get "now" once for S9 checks
    now_ms = (
        _safe_store_call(store, "current_ts_ms", signal.symbol, signal.timeframe)
        or _safe_store_call(store, "get_current_ts_ms", signal.symbol, signal.timeframe)
    )

    # --- 9.1 News / Volatility Calendar Veto ---
    if NewsFilter is not None and now_ms is not None:
        try:
            nf = NewsFilter(settings)
            if getattr(nf, "enabled", True) and nf.is_blocked(now_ms):
                reasons.append("NEWS_WINDOW")
        except Exception:
            pass  # ignore news failures

    # --- 9.2 Depth Thinness / Spread Check (multi-exchange placeholder) ---
    # Try to get richer top-of-book if the store exposes it; else fallback to the earlier book_ticker.
    bid_qty = None
    ask_qty = None
    bt_top = _safe_store_call(store, "get_book_top", signal.symbol)
    if isinstance(bt_top, dict):
        # Expecting keys like 'bid','ask','bid_qty','ask_qty' or shorthand 'b','a','B','A'
        b = bt_top.get("bid", bt_top.get("b"))
        a = bt_top.get("ask", bt_top.get("a"))
        Bq = bt_top.get("bid_qty", bt_top.get("B"))
        Aq = bt_top.get("ask_qty", bt_top.get("A"))
        if b is not None and a is not None:
            bid, ask = float(b), float(a)
        if Bq is not None:
            bid_qty = float(Bq)
        if Aq is not None:
            ask_qty = float(Aq)

    # If still None, try a tuple-style top with qtys in positions 2/3
    if bid_qty is None or ask_qty is None:
        if isinstance(book_ticker, tuple) and len(book_ticker) >= 4:
            try:
                bid_qty = float(book_ticker[2])
                ask_qty = float(book_ticker[3])
            except Exception:
                bid_qty = ask_qty = None

    if DepthAggregator is not None:
        try:
            da = DepthAggregator(settings)
            # We can evaluate spread-based thinness even without qtys; qty check only if we have them.
            eval_bid_qty = float(bid_qty) if bid_qty is not None else 0.0
            eval_ask_qty = float(ask_qty) if ask_qty is not None else 0.0
            depth_metrics = da.evaluate(float(bid), float(ask), eval_bid_qty, eval_ask_qty)
            details["depth_spread_bps"] = depth_metrics.get("spread_bps")
            details["depth_top_qty"] = depth_metrics.get("top_qty")
            details["depth_is_thin"] = depth_metrics.get("is_thin")
            # 🔧 Relaxed default: do NOT enable depth-thin veto unless user explicitly enables it
            if bool(settings.get("veto", {}).get("enable_depth_thin_check", False)) and depth_metrics.get("is_thin", 0.0) >= 1.0:
                reasons.append("DEPTH_THIN")
        except Exception:
            pass  # ignore depth failures

    # --- 9.3 CVD Alignment (order-flow proxy from OHLCV) ---
    if CVDComputer is not None:
        try:
            cvd_cfg = settings.get("alpha", {}).get("cvd", {}) if isinstance(settings.get("alpha", {}), dict) else {}
            lb = int(cvd_cfg.get("lookback", 200))
            sw = int(cvd_cfg.get("slope_window", 20))
            slope_thr = float(cvd_cfg.get("slope_threshold", 0.1))  # 🔧 new threshold
            rows = _safe_store_call(store, "get_recent_ohlcv", signal.symbol, signal.timeframe, lb)
            if rows:
                cvd_vals = CVDComputer(lookback=lb, slope_window=sw).compute_proxy(rows)
                cvd_slope = cvd_vals.get("cvd_slope", 0.0)
                details["cvd_slope"] = cvd_slope
                # 🔧 Relaxed default: OFF unless explicitly enabled
                cvd_align_enabled = bool(settings.get("veto", {}).get("enable_cvd_alignment", False))
                if cvd_align_enabled:
                    # Only veto if slope is meaningfully against the trade direction
                    if signal.decision == "LONG" and cvd_slope < -slope_thr:
                        reasons.append("CVD_WEAK")
                    if signal.decision == "SHORT" and cvd_slope > slope_thr:
                        reasons.append("CVD_WEAK")
        except Exception:
            pass  # ignore cvd failures

    # --- 9.4 Liquidation Spike Contra Veto ---
    if LiqPulse is not None and now_ms is not None:
        try:
            if bool(settings.get("veto", {}).get("enable_liq_spike_contra", True)):
                lp = LiqPulse(settings)
                liq = lp.latest_spike_flag(signal.symbol, int(now_ms))
                details["liq_z"] = liq.get("liq_z", 0.0)
                if float(liq.get("liq_spike", 0.0)) >= 1.0:
                    reasons.append("LIQUIDATION_SPIKE")
        except Exception:
            pass  # ignore liq failures

    # If any of the reasons were triggered, veto the trade:
    if reasons:
        return FilterResult(False, reason=";".join(reasons), details=details)

    # ----------------- 7. PASSED -----------------
    return FilterResult(True)
