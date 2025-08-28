"""
Derivatives posture feature pack
- funding_actual_8h, funding_pred_next, mins_to_funding
- funding_z, funding_pctl
- oi_notional, oi_change_%, oi_z, oi_rate_per_min
- oi_taxonomy classification per bar
- deriv_posture_score composite and policy suggestion

This module is intentionally lightweight and defensive: it reads whatever
information is available from FeatureStore and the optional FundingProvider.
"""
from typing import Optional, Dict
import math
import statistics
import time

from ultra_signals.core.custom_types import DerivativesFeatures


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def compute_z(value: Optional[float], window: list) -> Optional[float]:
    try:
        if value is None or not window:
            return None
        vals = [v for v in window if v is not None]
        if not vals:
            return None
        mu = statistics.mean(vals)
        sigma = statistics.pstdev(vals)
        if sigma == 0:
            return 0.0
        return (value - mu) / sigma
    except Exception:
        return None


def pct_rank(value: Optional[float], window: list) -> Optional[float]:
    try:
        if value is None or not window:
            return None
        vals = sorted([v for v in window if v is not None])
        if not vals:
            return None
        # position 0..len-1
        pos = 0
        for v in vals:
            if value >= v:
                pos += 1
        return pos / len(vals)
    except Exception:
        return None


def classify_oi_taxonomy(price_now: float, oi_now: Optional[float], oi_prev: Optional[float]) -> Optional[str]:
    try:
        if oi_now is None or oi_prev is None or price_now is None:
            return None
        # oi change positive -> build-up
        oi_chg = oi_now - oi_prev
        if oi_chg == 0:
            return None
        # We cannot access historical price change here; caller should compute sign externally.
        # For convenience assume caller passes price_now and price_prev via price_diff in oi_prev (hack avoided)
        # Keep simple: caller should set oi_taxonomy externally when price movement known.
        return None
    except Exception:
        return None


def compute_derivatives_posture(feature_store, symbol: str, timestamp_ms: Optional[int] = None, window_minutes: int = 480) -> DerivativesFeatures:
    """
    Build derivatives posture features for a symbol at given time.

    - feature_store: FeatureStore instance
    - symbol: market symbol
    - timestamp_ms: epoch ms at which to evaluate (defaults to now)
    - window_minutes: history window for z-score / percentile (default 8h -> 480m)
    """
    now_ms = int(timestamp_ms or int(time.time() * 1000))
    df_w = int(window_minutes)

    # Initialize output
    out = DerivativesFeatures()

    # Funding: read funding history via FeatureStore bridge
    try:
        hist = feature_store.get_funding_rate_history(symbol) or []
    except Exception:
        hist = []
    # hist entries expected as list of {funding_rate, funding_time}
    # build recent trail (last N observations)
    funding_trail = [float(h.get('funding_rate') or 0.0) for h in hist if h.get('funding_rate') is not None]
    out.funding_trail = funding_trail
    out.funding_now = funding_trail[-1] if funding_trail else None
    # predicted next if provider offers (look for funding_pred or funding_next)
    try:
        fp = None
        if feature_store._funding_provider and hasattr(feature_store._funding_provider, 'get_predicted'):
            fp = feature_store._funding_provider.get_predicted(symbol)
        out.funding_pred_next = float(fp) if fp is not None else None
    except Exception:
        out.funding_pred_next = None
    # minutes to next funding via store helper
    try:
        out.mins_to_funding = feature_store.get_minutes_to_next_funding(symbol, now_ms)
    except Exception:
        out.mins_to_funding = None

    # Normalizations
    try:
        out.funding_z = compute_z(out.funding_now, funding_trail) if funding_trail else None
        out.funding_pctl = pct_rank(out.funding_now, funding_trail) if funding_trail else None
    except Exception:
        out.funding_z = out.funding_pctl = None

    # Open interest: try flow_metrics or feature_store helper
    oi_now = None
    oi_prev = None
    try:
        # prefer explicit flow_metrics in latest features
        feats = feature_store.get_latest_features(symbol, None)
        fm = feats.get('flow_metrics') if feats else None
        if fm is not None:
            oi_now = getattr(fm, 'oi', None) or getattr(fm, 'oi_notional', None) or None
            oi_prev = getattr(fm, 'oi_prev', None) or None
            oi_rate = getattr(fm, 'oi_rate', None) or None
            out.oi_rate_per_min = oi_rate
    except Exception:
        pass
    # fallback: read provider hooks if present
    try:
        if oi_now is None and hasattr(feature_store, 'get_oi_snapshot'):
            snap = feature_store.get_oi_snapshot(symbol)
            if snap and isinstance(snap, dict):
                oi_now = snap.get('oi') or snap.get('oi_notional')
                oi_prev = snap.get('oi_prev')
    except Exception:
        pass

    out.oi_notional = float(oi_now) if oi_now is not None else None
    out.oi_prev_notional = float(oi_prev) if oi_prev is not None else None
    try:
        if oi_now is not None and oi_prev is not None and oi_prev != 0:
            out.oi_change_pct = (oi_now - oi_prev) / float(oi_prev)
        else:
            out.oi_change_pct = None
    except Exception:
        out.oi_change_pct = None

    # compute oi_z using recent history if feature_store exposes a small history
    oi_window = []
    try:
        # Try reading last window_minutes from funding provider cache as proxy if no oi history
        if hasattr(feature_store, '_feature_cache'):
            # try to extract flow_metrics.oi across cached bars
            cache = getattr(feature_store, '_feature_cache', {})
            sym = cache.get(symbol, {})
            for tf, series in sym.items():
                for ts, entry in series.items():
                    try:
                        fm = entry.get('flow_metrics')
                        if fm is None:
                            continue
                        val = getattr(fm, 'oi', None) or getattr(fm, 'oi_notional', None)
                        if val is not None:
                            oi_window.append(float(val))
                    except Exception:
                        continue
    except Exception:
        oi_window = []

    try:
        out.oi_z = compute_z(out.oi_notional, oi_window) if oi_window else None
    except Exception:
        out.oi_z = None

    # Taxonomy: using price vs oi change. We'll try to read recent price change from OHLCV
    price_now = None
    price_prev = None
    try:
        df = feature_store.get_ohlcv(symbol, (feature_store._settings or {}).get('runtime', {}).get('primary_timeframe', '5m'))
        if df is not None and len(df) >= 2:
            price_now = float(df['close'].iloc[-1])
            price_prev = float(df['close'].iloc[-2])
    except Exception:
        price_now = price_prev = None

    oi_tax = None
    try:
        if out.oi_notional is not None and out.oi_prev_notional is not None and price_now is not None and price_prev is not None:
            oi_delta = out.oi_notional - out.oi_prev_notional
            price_delta = price_now - price_prev
            # classify per definition
            if price_delta > 0 and oi_delta > 0:
                oi_tax = 'new_longs'
            elif price_delta < 0 and oi_delta > 0:
                oi_tax = 'new_shorts'
            elif price_delta > 0 and oi_delta < 0:
                oi_tax = 'short_cover'
            elif price_delta < 0 and oi_delta < 0:
                oi_tax = 'long_liq'
            else:
                oi_tax = None
    except Exception:
        oi_tax = None
    out.oi_taxonomy = oi_tax

    # basis: attempt perp-spot basis via feature_store helper (assume store exposes if available)
    try:
        if hasattr(feature_store, 'get_basis_bps'):
            out.basis_bps = feature_store.get_basis_bps(symbol)
            # for z compute, try windowing from cache
            bwin = []
            cache = getattr(feature_store, '_feature_cache', {})
            sym = cache.get(symbol, {})
            for tf, series in sym.items():
                for ts, entry in series.items():
                    try:
                        b = entry.get('derivatives') and getattr(entry.get('derivatives'), 'basis_bps', None)
                        if b is not None:
                            bwin.append(float(b))
                    except Exception:
                        continue
            out.basis_z = compute_z(out.basis_bps, bwin) if bwin else None
    except Exception:
        out.basis_bps = out.basis_z = None

    # Simple composite posture score: weighted blend
    try:
        components = []
        weights = []
        # funding_z normalized to [-3,3] -> map to [0,1]
        if out.funding_z is not None:
            fz = max(min(out.funding_z, 3.0), -3.0)
            # positive funding_z indicates long bleeding (avoid longs)
            components.append((fz + 3.0) / 6.0)
            weights.append(0.4)
        if out.oi_z is not None:
            oz = max(min(out.oi_z, 3.0), -3.0)
            components.append((oz + 3.0) / 6.0)
            weights.append(0.35)
        if out.basis_z is not None:
            bz = max(min(out.basis_z, 3.0), -3.0)
            components.append((bz + 3.0) / 6.0)
            weights.append(0.15)
        # venue disagreement: best-effort: if flow_metrics present across tfs disagree -> flag
        vflag = 0
        try:
            # naive detection: check funding_trail variance across providers if present as dict; skip otherwise
            if isinstance(hist, dict):
                # hist per-venue expected
                vals = []
                for k, v in hist.items():
                    try:
                        vals.append(float(v[-1].get('funding_rate')))
                    except Exception:
                        continue
                if vals and (max(vals) - min(vals)) > 0.0005:
                    vflag = 1
        except Exception:
            vflag = 0
        out.venue_disagreement_flag = vflag

        if components and weights and len(components) == len(weights):
            # weighted average
            s = sum(c * w for c, w in zip(components, weights)) / sum(weights)
            out.deriv_posture_score = max(0.0, min(1.0, float(s)))
        else:
            out.deriv_posture_score = None
    except Exception:
        out.deriv_posture_score = None

    # Overheat flags & simple policy suggestion
    try:
        cfg = getattr(feature_store, '_settings', {}) or {}
        thresholds = cfg.get('derivatives', {}) or {}
        hi = float(thresholds.get('funding_overheat_pos', 0.001))
        lo = float(thresholds.get('funding_overheat_neg', -0.001))
        oi_hi_z = float(thresholds.get('oi_z_hi', 2.0))
        oi_lo_z = float(thresholds.get('oi_z_lo', -2.0))
        out.deriv_overheat_flag_long = 1 if (out.funding_now is not None and out.funding_now >= hi) or (out.oi_z is not None and out.oi_z >= oi_hi_z) else 0
        out.deriv_overheat_flag_short = 1 if (out.funding_now is not None and out.funding_now <= lo) or (out.oi_z is not None and out.oi_z <= oi_lo_z) else 0
        # Settlement window logic
        in_settlement = False
        try:
            mins = out.mins_to_funding
            pre = int(cfg.get('settlement', {}).get('pre_minutes', 30) or 30)
            post = int(cfg.get('settlement', {}).get('post_minutes', 15) or 15)
            if mins is not None and mins <= pre:
                in_settlement = True
        except Exception:
            in_settlement = False
        # Suggest policy: simple rule-set
        policy = 'allow'
        if in_settlement:
            policy = 'delay_to_post_settlement'
        if out.deriv_overheat_flag_long and out.deriv_posture_score and out.deriv_posture_score > 0.6:
            policy = 'veto' if out.deriv_overheat_flag_long else policy
        if out.deriv_overheat_flag_short and out.deriv_posture_score and out.deriv_posture_score > 0.6:
            policy = 'veto' if out.deriv_overheat_flag_short else policy
        out.policy_suggest = policy
    except Exception:
        out.policy_suggest = None

    return out
