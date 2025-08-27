"""Liquidity & Micro-Regime Gate (Sprint 29).

Evaluates microstructure health (spread, depth/impact, imbalance+volatility,
whipsaw noise) and returns an action: VETO, DAMPEN, NONE.

This module is intentionally dependency-light; it only consumes a
`BookHealth` snapshot or proxy metrics. All thresholds come from
settings['micro_liquidity'] with per-regime profile sections.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import time
from loguru import logger

try:  # runtime optional import (unit tests can stub BookHealth objects)
    from ultra_signals.market.book_health import BookHealth
except Exception:  # pragma: no cover
    BookHealth = object  # type: ignore


@dataclass(slots=True)
class LiquidityGateDecision:
    action: str                # VETO | DAMPEN | NONE
    reason: Optional[str] = None
    size_mult: Optional[float] = None
    widen_stop_mult: Optional[float] = None
    maker_only: bool = False
    meta: Dict[str, Any] = None


class LiquidityGate:
    def __init__(self, settings: Dict):
        self.settings = settings or {}
        self._last_veto_ts: Dict[str, int] = {}  # symbol -> epoch seconds

    # public evaluate -----------------------------------------------------
    def evaluate(self, symbol: str, now_ts: int, profile: str, book: Any) -> LiquidityGateDecision:
        cfg = (self.settings.get("micro_liquidity") or {}) if isinstance(self.settings, dict) else {}
        if not cfg.get("enabled", True):
            return LiquidityGateDecision(action="NONE", meta={"disabled": True})
        prof_cfg = ((cfg.get("profiles") or {}).get(profile) or (cfg.get("profiles") or {}).get("trend") or {})
        missing_policy = str(cfg.get("missing_feed_policy", "SAFE")).upper()
        cooldown_secs = int(cfg.get("cooldown_after_veto_secs", 0))

        # Cooldown check
        last_veto = self._last_veto_ts.get(symbol, 0)
        if cooldown_secs > 0 and (now_ts - last_veto) < cooldown_secs:
            return LiquidityGateDecision(action="VETO", reason="COOLDOWN", meta={"cooldown": True})

        if book is None:
            if missing_policy == "OFF":
                return LiquidityGateDecision(action="NONE", reason="NO_FEED", meta={"policy": "OFF"})
            if missing_policy == "OPEN":
                return LiquidityGateDecision(action="NONE", reason="NO_FEED", meta={"policy": "OPEN"})
            # SAFE -> DAMPEN
            damp = prof_cfg.get("dampen", {})
            return LiquidityGateDecision(action="DAMPEN", reason="MISSING_FEED", size_mult=damp.get("size_mult"), widen_stop_mult=damp.get("widen_stop_mult"), maker_only=bool(damp.get("maker_only", False)), meta={"policy": "SAFE"})

        # Metrics extraction (allow None -> skip)
        spread_bps = getattr(book, "spread_bps", None)
        impact_50k = getattr(book, "impact_50k", None)
        dr = getattr(book, "dr", None)
        rv_5s = getattr(book, "rv_5s", None)
        mt = getattr(book, "mt", None)

        # Thresholds
        spread_cap = float(prof_cfg.get("spread_cap_bps", 9.9))
        spread_warn = float(prof_cfg.get("spread_warn_bps", spread_cap * 0.75))
        impact_cap = float(prof_cfg.get("impact_cap_bps", 25.0))
        impact_warn = float(prof_cfg.get("impact_warn_bps", impact_cap * 0.6))
        rv_cap = float(prof_cfg.get("rv_cap_bps", 15.0))
        rv_whip_cap = float(prof_cfg.get("rv_whip_cap_bps", rv_cap * 1.5))
        dr_skew_cap = float(prof_cfg.get("dr_skew_cap", 0.75))
        mt_trend_min = float(prof_cfg.get("mt_trend_min", 0.15))
        damp_cfg = prof_cfg.get("dampen", {})

        # 1) Max spread
        if spread_bps is not None and spread_bps > spread_cap:
            self._last_veto_ts[symbol] = now_ts
            return LiquidityGateDecision(action="VETO", reason="WIDE_SPREAD", meta={"spread_bps": spread_bps})
        # 2) Thin depth / high impact
        if impact_50k is not None and impact_50k > impact_cap:
            self._last_veto_ts[symbol] = now_ts
            return LiquidityGateDecision(action="VETO", reason="THIN_BOOK", meta={"impact_50k": impact_50k})
        # 3) Imbalance + high vol = spoofy
        if dr is not None and rv_5s is not None and abs(dr) > dr_skew_cap and rv_5s > rv_cap:
            self._last_veto_ts[symbol] = now_ts
            return LiquidityGateDecision(action="VETO", reason="SPOOFY", meta={"dr": dr, "rv_5s": rv_5s})
        # 4) Whipsaw noise: elevated rv + no trendiness
        if rv_5s is not None and rv_5s > rv_whip_cap and (mt is None or abs(mt) < mt_trend_min):
            self._last_veto_ts[symbol] = now_ts
            return LiquidityGateDecision(action="VETO", reason="CHAOTIC", meta={"rv_5s": rv_5s, "mt": mt})
        # 5) Dampen path
        if (spread_bps is not None and spread_bps > spread_warn) or (impact_50k is not None and impact_50k > impact_warn):
            return LiquidityGateDecision(
                action="DAMPEN",
                reason="WARN",
                size_mult=damp_cfg.get("size_mult"),
                widen_stop_mult=damp_cfg.get("widen_stop_mult"),
                maker_only=bool(damp_cfg.get("maker_only", False)),
                meta={"spread_bps": spread_bps, "impact_50k": impact_50k},
            )
        return LiquidityGateDecision(action="NONE", meta={"spread_bps": spread_bps, "impact_50k": impact_50k, "rv_5s": rv_5s})


def evaluate_gate(symbol: str, now_ts: int, profile: str, book: Any, settings: Dict, gate: LiquidityGate | None = None) -> LiquidityGateDecision:
    """Convenience wrapper. If a gate instance is provided reuse it, else build a fresh one.

    A fresh instance each call keeps tests deterministic (no shared cooldown)
    while production callers can pass a persistent LiquidityGate for stateful
    cooldown tracking.
    """
    gate_obj = gate or LiquidityGate(settings)
    try:
        out = gate_obj.evaluate(symbol, now_ts, profile, book)
        # Telemetry cache (best-effort, ignore errors). Avoid import cycles.
        try:
            globals().setdefault('_LAST_LQ', {})[symbol] = {
                'ts': now_ts,
                'symbol': symbol,
                'profile': profile,
                'action': out.action,
                'reason': out.reason,
                'meta': out.meta,
                'size_mult': out.size_mult,
            }
        except Exception:
            pass
        return out
    except Exception as e:  # pragma: no cover
        logger.debug(f"Liquidity gate evaluate error: {e}")
        return LiquidityGateDecision(action="NONE", reason="ERR")

__all__ = ["LiquidityGateDecision", "evaluate_gate", "LiquidityGate"]
