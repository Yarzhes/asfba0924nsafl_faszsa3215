""""
Risk Management Filters

Provides the apply_filters function to check if a signal passes basic risk filters.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from loguru import logger

from ultra_signals.engine.confluence import confluence_htf_agrees
from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import Signal
import time
from collections import defaultdict, deque

# Sniper mode enforcement
try:
    from .sniper_counters import get_sniper_counters, reset_sniper_counters
except ImportError:
    get_sniper_counters = reset_sniper_counters = None

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

# Sprint 40 sentiment (optional import)
try:
    from ultra_signals.sentiment import SentimentEngine  # feature_view(), maybe_veto()
except Exception:  # pragma: no cover
    SentimentEngine = None  # type: ignore


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


def apply_filters(signal: Signal, store: FeatureStore, settings: dict, metrics: Optional[Any] = None, *, record_candidate: bool = True) -> FilterResult:
    """
    Checks if a signal passes basic risk filters.
    - Must check: warmup bars and spread (bid/ask) availability.
    - `store` is FeatureStore, use its stored OHLCV and book ticker.
    - `settings` is a dict (pydantic model dumped by .model_dump()).
    """
    # Metrics: every invocation counts as a candidate
    if record_candidate:
        try:
            if metrics and hasattr(metrics, 'record_candidate'):
                metrics.record_candidate()
        except Exception:
            pass

    # ----------------- 1. WARMUP CHECK -----------------
    try:
        warmup_periods = settings.get("features", {}).get("warmup_periods", 20)
        available_bars = store.get_warmup_status(signal.symbol, signal.timeframe)
        if available_bars < warmup_periods:
            logger.info(f"[RiskFilters] WARMUP_INCOMPLETE for {signal.symbol} {signal.timeframe}: available={available_bars}, required={warmup_periods}")
            fr = FilterResult(False, reason="WARMUP_INCOMPLETE")
            logger.debug(f"[RiskFilters] BLOCK {signal.symbol} {signal.timeframe} reason={fr.reason}")
            try:
                if metrics and hasattr(metrics, 'record_block'):
                    metrics.record_block(fr.reason)
            except Exception:
                pass
            return fr
    except Exception as e:
        logger.error(f"[RiskFilters] Error in WARMUP CHECK for {signal.symbol} {signal.timeframe}: {e}")
        return FilterResult(False, reason=f"FILTER_ERR_WARMUP:{e}")

    # ----------------- 2. BOOK TICKER VALIDATION -----------------
    try:
        book_ticker = store.get_book_ticker(signal.symbol)
        if not book_ticker:
            fr = FilterResult(False, reason="MISSING_BOOK_TICKER")
            logger.debug(f"[RiskFilters] BLOCK {signal.symbol} {signal.timeframe} reason={fr.reason}")
            try:
                if metrics and hasattr(metrics, 'record_block'):
                    metrics.record_block(fr.reason)
            except Exception:
                pass
            return fr

        # Ensure book_ticker is a tuple with at least 2 elements
        if not isinstance(book_ticker, tuple) or len(book_ticker) < 2:
            fr = FilterResult(False, reason="INVALID_BOOK_TICKER")
            logger.debug(f"[RiskFilters] BLOCK {signal.symbol} {signal.timeframe} reason={fr.reason} book_ticker={book_ticker}")
            try:
                if metrics and hasattr(metrics, 'record_block'):
                    metrics.record_block(fr.reason)
            except Exception:
                pass
            return fr

        bid, ask = book_ticker[:2]
        if bid <= 0 or ask <= 0:
            fr = FilterResult(False, reason="INVALID_PRICE")
            logger.debug(f"[RiskFilters] BLOCK {signal.symbol} {signal.timeframe} reason={fr.reason} bid={bid} ask={ask}")
            try:
                if metrics and hasattr(metrics, 'record_block'):
                    metrics.record_block(fr.reason)
            except Exception:
                pass
            return fr

        mid_price = (ask + bid) / 2
        if mid_price == 0:
            fr = FilterResult(False, reason="ZERO_MID_PRICE")
            logger.debug(f"[RiskFilters] BLOCK {signal.symbol} {signal.timeframe} reason={fr.reason} bid={bid} ask={ask}")
            try:
                if metrics and hasattr(metrics, 'record_block'):
                    metrics.record_block(fr.reason)
            except Exception:
                pass
            return fr
    except Exception as e:
        logger.error(f"[RiskFilters] Error in BOOK TICKER VALIDATION for {signal.symbol} {signal.timeframe}: {e}")
        return FilterResult(False, reason=f"FILTER_ERR_BOOK_TICKER:{e}")

    # ----------------- 3. SPREAD CALCULATION -----------------
    try:
        spread = ask - bid
        spread_pct = spread / mid_price
    except Exception as e:
        logger.error(f"[RiskFilters] Error in SPREAD CALCULATION for {signal.symbol} {signal.timeframe}: {e}")
        return FilterResult(False, reason=f"FILTER_ERR_SPREAD_CALC:{e}")

    # ----------------- 4. MAX SPREAD FETCH (UPDATED) -----------------
    try:
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
    except Exception as e:
        logger.error(f"[RiskFilters] Error in MAX SPREAD FETCH for {signal.symbol} {signal.timeframe}: {e}")
        return FilterResult(False, reason=f"FILTER_ERR_MAX_SPREAD_FETCH:{e}")

    # ----------------- 5. APPLY SPREAD % FILTER -----------------
    try:
        if spread_pct > max_spread_pct:
            fr = FilterResult(False, reason="SPREAD_TOO_WIDE", details={"spread": spread_pct, "max_allowed": max_spread_pct})
            logger.debug(f"[RiskFilters] BLOCK {signal.symbol} {signal.timeframe} reason={fr.reason} spread_pct={spread_pct:.6f} max_allowed={max_spread_pct}")
            try:
                if metrics and hasattr(metrics, 'record_block'):
                    metrics.record_block(fr.reason)
            except Exception:
                pass
            return fr
    except Exception as e:
        logger.error(f"[RiskFilters] Error in APPLY SPREAD % FILTER for {signal.symbol} {signal.timeframe}: {e}")
        return FilterResult(False, reason=f"FILTER_ERR_APPLY_SPREAD_FILTER:{e}")

    # =================================================================
    # ============= SPRINT 8 ADDITIONS (OPTIONAL, NON-BREAKING) ========
    # =================================================================
    # We'll collect additional gating reasons. If any are present, we block.
    reasons: List[str] = []
    details: Dict[str, Any] = {
        "spread_pct": spread_pct
    }

    # --- 6.1 ATR Percentile Gate ---
    try:
        # Only trade if ATR percentile is high enough (volatility present).
        atr_gate_pct = int(settings.get("filters", {}).get("atr_gate_pct", 0))  # default 0 = disabled
        atr_pct = _safe_store_call(store, "get_atr_percentile", signal.symbol, signal.timeframe)
        details["atr_percentile"] = atr_pct
        if atr_gate_pct > 0 and atr_pct is not None and atr_pct < atr_gate_pct:
            reasons.append("LOW_ATR")
    except Exception as e:
        logger.error(f"[RiskFilters] Error in ATR Percentile Gate for {signal.symbol} {signal.timeframe}: {e}")
        reasons.append(f"FILTER_ERR_ATR_PERCENTILE:{e}")

    # --- 6.2 ADX Trend Strength Gate ---
    try:
        adx_min = int(settings.get("filters", {}).get("adx_min", 0))  # default 0 = disabled
        adx_val = _safe_store_call(store, "get_adx", signal.symbol, signal.timeframe)
        details["adx"] = adx_val
        if adx_min > 0 and adx_val is not None and adx_val < adx_min:
            reasons.append("LOW_ADX")
    except Exception as e:
        logger.error(f"[RiskFilters] Error in ADX Trend Strength Gate for {signal.symbol} {signal.timeframe}: {e}")
        reasons.append(f"FILTER_ERR_ADX_TREND:{e}")

    # --- 6.3 TR Compression (skip ultra-chop) ---
    try:
        tr_comp_max = settings.get("filters", {}).get("tr_compression_max", None)
        tr_comp = _safe_store_call(store, "get_tr_compression", signal.symbol, signal.timeframe)
        details["tr_compression"] = tr_comp
        if tr_comp_max is not None and tr_comp is not None:
            try:
                tr_comp_max_f = float(tr_comp_max)
                if tr_comp <= tr_comp_max_f:
                    reasons.append("TR_COMPRESSION")
            except Exception as e:
                logger.error(f"[RiskFilters] Error in TR Compression (inner) for {signal.symbol} {signal.timeframe}: {e}")
                reasons.append(f"FILTER_ERR_TR_COMPRESSION_INNER:{e}")
    except Exception as e:
        logger.error(f"[RiskFilters] Error in TR Compression (outer) for {signal.symbol} {signal.timeframe}: {e}")
        reasons.append(f"FILTER_ERR_TR_COMPRESSION_OUTER:{e}")

    # --- 6.4 Funding Window Avoidance (Sprint 8 baseline) ---
    try:
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
                    mins_to_funding = fp.minutes_to_next(signal.symbol)
            except Exception as e:
                logger.error(f"[RiskFilters] Error in Funding Window Avoidance (inner) for {signal.symbol} {signal.timeframe}: {e}")
                mins_to_funding = None

        details["mins_to_funding"] = mins_to_funding
        if window_min > 0 and mins_to_funding is not None and abs(mins_to_funding) < window_min:
            reasons.append("NEAR_FUNDING_WINDOW")
    except Exception as e:
        logger.error(f"[RiskFilters] Error in Funding Window Avoidance (outer) for {signal.symbol} {signal.timeframe}: {e}")
        reasons.append(f"FILTER_ERR_FUNDING_WINDOW:{e}")

    # --- 6.5 Wide Spread (bps) Veto (separate from % check above) ---
    try:
        wide_spread_bps = int(settings.get("veto", {}).get("wide_spread_bps", 0))  # 0 = disabled
        # If your store can provide spread in basis points:
        spread_bps = _safe_store_call(store, "get_spread_bps", signal.symbol)
        details["spread_bps"] = spread_bps
        if wide_spread_bps > 0 and spread_bps is not None and spread_bps > wide_spread_bps:
            reasons.append("WIDE_SPREAD")
    except Exception as e:
        logger.error(f"[RiskFilters] Error in Wide Spread (bps) Veto for {signal.symbol} {signal.timeframe}: {e}")
        reasons.append(f"FILTER_ERR_WIDE_SPREAD_BPS:{e}")

    # --- 6.6 Multi-Timeframe Confluence ---
    try:
        # Require HTF regime alignment if enabled.
        confluence_required = bool(settings.get("confluence", {}).get("require_regime_align", True))
        if confluence_required:
            if not _htf_confluence_agrees(signal, store, settings):
                reasons.append("MTF_DISAGREE")
    except Exception as e:
        logger.error(f"[RiskFilters] Error in Multi-Timeframe Confluence for {signal.symbol} {signal.timeframe}: {e}")
        reasons.append(f"FILTER_ERR_MTF_CONFLUENCE:{e}")

    # =================================================================
    # ======================= SPRINT 9 VETO BLOCKS =====================
    # =================================================================

    # Helper to get "now" once for S9 checks
    now_ms = (
        _safe_store_call(store, "current_ts_ms", signal.symbol, signal.timeframe)
        or _safe_store_call(store, "get_current_ts_ms", signal.symbol, signal.timeframe)
    )

    # --- 9.1 News / Volatility Calendar Veto ---
    try:
        if NewsFilter is not None and now_ms is not None:
            try:
                nf = NewsFilter(settings)
                if getattr(nf, "enabled", True) and nf.is_blocked(now_ms):
                    reasons.append("NEWS_WINDOW")
            except Exception as e:
                logger.error(f"[RiskFilters] Error in News / Volatility Calendar Veto (inner) for {signal.symbol} {signal.timeframe}: {e}")
                pass  # ignore news failures
    except Exception as e:
        logger.error(f"[RiskFilters] Error in News / Volatility Calendar Veto (outer) for {signal.symbol} {signal.timeframe}: {e}")
        reasons.append(f"FILTER_ERR_NEWS_VETO:{e}")

    # --- 9.2 Depth Thinness / Spread Check (multi-exchange placeholder) ---
    try:
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
                except Exception as e:
                    logger.error(f"[RiskFilters] Error in Depth Thinness / Spread Check (inner tuple) for {signal.symbol} {signal.timeframe}: {e}")
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
            except Exception as e:
                logger.error(f"[RiskFilters] Error in Depth Thinness / Spread Check (inner DepthAggregator) for {signal.symbol} {signal.timeframe}: {e}")
                pass  # ignore depth failures
    except Exception as e:
        logger.error(f"[RiskFilters] Error in Depth Thinness / Spread Check (outer) for {signal.symbol} {signal.timeframe}: {e}")
        reasons.append(f"FILTER_ERR_DEPTH_THINNESS:{e}")

    # --- 9.3 CVD Alignment (order-flow proxy from OHLCV) ---
    try:
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
            except Exception as e:
                logger.error(f"[RiskFilters] Error in CVD Alignment (inner) for {signal.symbol} {signal.timeframe}: {e}")
                pass  # ignore cvd failures
    except Exception as e:
        logger.error(f"[RiskFilters] Error in CVD Alignment (outer) for {signal.symbol} {signal.timeframe}: {e}")
        reasons.append(f"FILTER_ERR_CVD_ALIGNMENT:{e}")

    # --- 9.4 Liquidation Spike Contra Veto ---
    try:
        if LiqPulse is not None and now_ms is not None:
            try:
                if bool(settings.get("veto", {}).get("enable_liq_spike_contra", True)):
                    lp = LiqPulse(settings)
                    liq = lp.latest_spike_flag(signal.symbol, int(now_ms))
                    details["liq_z"] = liq.get("liq_z", 0.0)
                    if float(liq.get("liq_spike", 0.0)) >= 1.0:
                        reasons.append("LIQUIDATION_SPIKE")
            except Exception as e:
                logger.error(f"[RiskFilters] Error in Liquidation Spike Contra Veto (inner) for {signal.symbol} {signal.timeframe}: {e}")
                pass  # ignore liq failures
    except Exception as e:
        logger.error(f"[RiskFilters] Error in Liquidation Spike Contra Veto (outer) for {signal.symbol} {signal.timeframe}: {e}")
        reasons.append(f"FILTER_ERR_LIQUIDATION_SPIKE:{e}")

    # If any of the reasons were triggered, veto the trade:
    if reasons:
        fr = FilterResult(False, reason=";".join(reasons), details=details)
        logger.debug(f"[RiskFilters] BLOCK {signal.symbol} {signal.timeframe} reason={fr.reason} details={details}")
        try:
            if metrics and hasattr(metrics, 'record_block'):
                metrics.record_block(fr.reason)
        except Exception:
            pass
        return fr

    # ----------------- 10. SNIPER MODE ENFORCEMENT -----------------
    try:
        # Enforce runtime.sniper_mode caps (per-hour, daily) and optional MTF confirm requirement.
        rt = settings.get('runtime', {}) or {}
        sniper = rt.get('sniper_mode') or {}
        if sniper and sniper.get('enabled'):
            # require MTF confirm if configured: earlier we append 'MTF_DISAGREE' to reasons when HTF fails
            if sniper.get('mtf_confirm', False):
                # If confluence check above already added MTF_DISAGREE we would have returned; but some callers only flag it in vote_detail.
                # To be conservative, re-check HTF confluence explicitly here and block if disagree.
                if not _htf_confluence_agrees(signal, store, settings):
                    fr = FilterResult(False, reason='SNIPER_MTF_REQUIRED')
                    try:
                        if metrics and hasattr(metrics, 'record_block'):
                            metrics.record_block(fr.reason)
                    except Exception:
                        pass
                    return fr

            # Use Redis-backed counters if available, fallback to in-memory
            # Aug 29 2025: Hourly sniper cap fully disabled per user request.
            # We still honor daily_cap if provided, but ignore any max_signals_per_hour setting.
            _orig_hourly = sniper.get('max_signals_per_hour')
            max_per_hour = 0  # force-disable hourly cap
            daily_cap = int(sniper.get('daily_signal_cap') or 0)
            
            if (max_per_hour > 0 or daily_cap > 0) and get_sniper_counters:
                counters = get_sniper_counters(settings)
                block_reason = counters.check_and_increment(max_per_hour, daily_cap)
                if block_reason:
                    fr = FilterResult(False, reason=block_reason)
                    logger.debug(f"[RiskFilters] BLOCK {signal.symbol} {signal.timeframe} reason={fr.reason}")
                    try:
                        if metrics and hasattr(metrics, 'record_block'):
                            metrics.record_block(fr.reason)
                    except Exception:
                        pass
                    return fr
            
            # Legacy in-memory fallback if sniper_counters unavailable
            elif max_per_hour > 0 or daily_cap > 0:
                now = int(time.time())
                
                # Initialize simple module-level caches if missing (single deque per window)
                if not hasattr(apply_filters, '_sniper_history'):
                    apply_filters._sniper_history = {'hour': deque(), 'day': deque()}

                def _clear_history():
                    apply_filters._sniper_history = {'hour': deque(), 'day': deque()}
                apply_filters._sniper_history_clear = _clear_history

                hist = apply_filters._sniper_history
                h_deque = hist['hour']
                d_deque = hist['day']

                # purge old timestamps outside windows
                while h_deque and h_deque[0] < now - 3600:
                    h_deque.popleft()
                while d_deque and d_deque[0] < now - 86400:
                    d_deque.popleft()

                # Hourly cap disabled: we no longer block on h_deque length
                if daily_cap > 0 and len(d_deque) >= daily_cap:
                    fr = FilterResult(False, reason='SNIPER_DAILY_CAP')
                    logger.debug(f"[RiskFilters] BLOCK {signal.symbol} {signal.timeframe} reason={fr.reason}")
                    try:
                        if metrics and hasattr(metrics, 'record_block'):
                            metrics.record_block(fr.reason)
                    except Exception:
                        pass
                    return fr

                # If allowed, record this planned signal timestamp so daily counting works.
                now_ts = int(time.time())
                # Still append to hour deque (no blocking) to keep historical structure intact
                h_deque.append(now_ts)
                d_deque.append(now_ts)
                # Keep deques bounded (slightly larger than caps to avoid unbounded growth)
                max_keep = max(max_per_hour * 2 if max_per_hour>0 else 100, 100)
                while len(h_deque) > max_keep:
                    h_deque.popleft()
                while len(d_deque) > max_keep * 24:
                    d_deque.popleft()
    except Exception as e:
        logger.error(f"[RiskFilters] Error in SNIPER MODE ENFORCEMENT for {signal.symbol} {signal.timeframe}: {e}")
        reasons.append(f"FILTER_ERR_SNIPER_MODE:{e}")

    # ======================= SPRINT 40 SENTIMENT VETO =====================
    # Evaluate sentiment extreme veto only if earlier filters passed.
    try:
        sent_cfg = settings.get("sentiment") if isinstance(settings.get("sentiment"), dict) else None
        if sent_cfg and sent_cfg.get("enabled", True) and sent_cfg.get("veto_extremes", True) and SentimentEngine is not None:
            # Lazy singleton: attach to settings dict to avoid repeated init cost.
            _sent_engine = sent_cfg.get("_engine_instance")
            if _sent_engine is None:
                try:
                    _sent_engine = SentimentEngine(settings)
                    sent_cfg["_engine_instance"] = _sent_engine  # type: ignore
                except Exception as e:
                    logger.error(f"[RiskFilters] Error initializing SentimentEngine for {signal.symbol} {signal.timeframe}: {e}")
                    _sent_engine = None
            if _sent_engine is not None:
                # Non-blocking step: attempt to refresh in a lightweight way.
                try:
                    _sent_engine.step()
                except Exception as e:
                    logger.error(f"[RiskFilters] Error in SentimentEngine step for {signal.symbol} {signal.timeframe}: {e}")
                    pass
                veto_reason = _sent_engine.maybe_veto(signal.symbol)
                if veto_reason:
                    fr = FilterResult(False, reason=veto_reason, details=details)
                    logger.debug(f"[RiskFilters] BLOCK {signal.symbol} {signal.timeframe} reason={fr.reason} details={details}")
                    try:
                        if metrics and hasattr(metrics, 'record_block'):
                            metrics.record_block(fr.reason)
                    except Exception:
                        pass
                    return fr
    except Exception as e:
        logger.error(f"[RiskFilters] Error in SENTIMENT VETO (outer) for {signal.symbol} {signal.timeframe}: {e}")
        reasons.append(f"FILTER_ERR_SENTIMENT_VETO:{e}")

    # ----------------- 7. PASSED -----------------
    # =============== META-SCORER / EXPECTANCY GATING (optional) ==================
    try:
        ms_cfg = settings.get('meta_scorer') if isinstance(settings.get('meta_scorer'), dict) else None
        if ms_cfg:
            # Expect the caller to attach calibrated probability & expectancy to signal (if available)
            p_win = getattr(signal, 'confidence', None)
            entropy = getattr(signal, 'entropy', None)
            exp_R = None
            try:
                vd = signal.vote_detail or {}
                exp_R = vd.get('expected_R')
            except Exception as e:
                logger.error(f"[RiskFilters] Error getting expected_R from signal for {signal.symbol} {signal.timeframe}: {e}")
                exp_R = None
            # Safely extract numeric values from config, handling dict/object cases
            try:
                p_min_raw = ms_cfg.get('p_win_min', 0.0)
                p_min_global = float(p_min_raw) if isinstance(p_min_raw, (int, float, str)) else 0.0
            except (TypeError, ValueError):
                p_min_global = 0.0
                
            try:
                entropy_raw = ms_cfg.get('entropy_max', 1.0)
                entropy_max = float(entropy_raw) if isinstance(entropy_raw, (int, float, str)) else 1.0
            except (TypeError, ValueError):
                entropy_max = 1.0
                
            try:
                min_exp_R_raw = ms_cfg.get('min_expected_R', 0.0)
                min_exp_R = float(min_exp_R_raw) if isinstance(min_exp_R_raw, (int, float, str)) else 0.0
            except (TypeError, ValueError):
                min_exp_R = 0.0
            # Regime-specific override
            try:
                regime_over = ms_cfg.get('p_win_min_by_regime', {}) or {}
                regime_label = None
                if isinstance(signal.vote_detail, dict):
                    rg = signal.vote_detail.get('regime') or {}
                    if isinstance(rg, dict):
                        regime_label = rg.get('regime_label') or rg.get('label')
                if regime_label and regime_label in regime_over:
                    p_min_global = float(regime_over.get(regime_label, p_min_global))
            except Exception as e:
                logger.error(f"[RiskFilters] Error in Regime-specific override for {signal.symbol} {signal.timeframe}: {e}")
                pass
            meta_reasons = []
            if p_win is not None and p_win < p_min_global:
                meta_reasons.append('META_PWIN_LOW')
            if entropy is not None and entropy > entropy_max:
                meta_reasons.append('META_ENTROPY_HIGH')
            if exp_R is not None and exp_R < min_exp_R:
                meta_reasons.append('META_EXPECTED_R_LOW')
            if meta_reasons:
                fr = FilterResult(False, reason=';'.join(meta_reasons))
                try:
                    if metrics and hasattr(metrics, 'record_block'):
                        metrics.record_block(fr.reason)
                except Exception:
                    pass
                return fr
    except Exception as e:
        logger.error(f"[RiskFilters] Error in META-SCORER / EXPECTANCY GATING for {signal.symbol} {signal.timeframe}: {e}")
        reasons.append(f"FILTER_ERR_META_SCORER:{e}")

    try:
        if metrics and hasattr(metrics, 'record_allowed'):
            metrics.record_allowed()
    except Exception:
        pass
    return FilterResult(True)
