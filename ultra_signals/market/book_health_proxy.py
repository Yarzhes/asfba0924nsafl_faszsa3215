"""Backtest / historical proxy computations for BookHealth (Sprint 29).

Used when full L2 orderbook not available. Functions accept a lightweight
feature dict (e.g. last bar OHLCV + volatility+volume features) and return
synthetic approximations used by the Liquidity Gate. All outputs are floats
in basis points where noted.
"""
from __future__ import annotations
from typing import Dict, Optional


def compute_proxies(feats: Dict) -> Dict[str, Optional[float]]:
    ohlcv = feats.get("ohlcv", {}) if isinstance(feats, dict) else {}
    close = float(ohlcv.get("close", 0.0) or 0.0)
    high = float(ohlcv.get("high", close) or close)
    low = float(ohlcv.get("low", close) or close)
    vol_block = feats.get("volatility") or {}
    vol_flow = feats.get("volume_flow") or {}
    bb_up = getattr(vol_block, "bbands_upper", None) if vol_block else None
    bb_lo = getattr(vol_block, "bbands_lower", None) if vol_block else None
    atr = getattr(vol_block, "atr", None) if vol_block else None
    vwap = getattr(vol_flow, "vwap", None) if vol_flow else None
    vol_z = getattr(vol_flow, "volume_z_score", None) if vol_flow else None

    spread_bps_proxy = None
    impact_50k_proxy = None
    dr_proxy = None
    rv_5s_proxy = None

    try:
        if close and bb_up and bb_lo:
            bb_width = (bb_up - bb_lo) / close * 10_000 if close else 0.0
            atr_pct = (atr / close * 10_000) if atr and close else 0.0
            spread_bps_proxy = max(bb_width * 0.05, atr_pct * 0.25)  # coarse scaling
    except Exception:
        pass
    try:
        bar_range_bps = (high - low) / close * 10_000 if close else None
        if bar_range_bps is not None:
            denom = (abs(vol_z) + 1.5) if vol_z is not None else 2.0
            impact_50k_proxy = bar_range_bps / denom
    except Exception:
        pass
    try:
        if vwap and bb_up and bb_lo and close:
            half_bw = (bb_up - bb_lo) / 2.0
            if half_bw > 0:
                dr_proxy_raw = (close - vwap) / half_bw
                # clamp to [-1,1]
                if dr_proxy_raw > 1: dr_proxy_raw = 1
                if dr_proxy_raw < -1: dr_proxy_raw = -1
                dr_proxy = dr_proxy_raw
    except Exception:
        pass
    try:
        if high and low and close:
            rv_5s_proxy = min((high - low) / close * 10_000, 50.0)
    except Exception:
        pass
    return {
        "spread_bps": spread_bps_proxy,
        "impact_50k": impact_50k_proxy,
        "dr": dr_proxy,
        "rv_5s": rv_5s_proxy,
    }

__all__ = ["compute_proxies"]
