from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import time

from ultra_signals.core.custom_types import (
    RegimeFeatures,
    RegimeMode,
    RegimeProfile,
    VolatilityBucket,
    VolState,
    NewsState,
    LiquidityState,
)

@dataclass
class _Counters:
    trend: int = 0
    mean_revert: int = 0
    chop: int = 0

@dataclass
class RegimeStateMachine:
    """Small persistent state machine with hysteresis + cooldown.

    Enhancements (Sprint 10 final):
    - Tracks total cooldown (for ribbon logging denominator)
    - Exit thresholds respected (won't flip until current regime exit conditions met)
    """

    current: RegimeMode = RegimeMode.CHOP
    counters: _Counters = field(default_factory=_Counters)
    since_ts: int = 0
    last_flip_ts: int = 0
    cooldown_left: int = 0
    cooldown_total: int = 0

    def update(
        self,
        candidate: RegimeMode,
        hysteresis_hits: int,
        cooldown_bars: int,
        ts_bar: int,
        allow_override: bool,
        strong_flag: bool,
    ) -> RegimeMode:
        # Cooldown freeze unless strong override permitted
        if self.cooldown_left > 0 and candidate != self.current:
            if not (allow_override and strong_flag):
                self.cooldown_left -= 1
                return self.current

        # Increment candidate counter / decay others (soft hysteresis)
        for name in ["trend", "mean_revert", "chop"]:
            val = getattr(self.counters, name)
            if name == candidate.value:
                setattr(self.counters, name, min(hysteresis_hits, val + 1))
            else:
                setattr(self.counters, name, max(0, val - 1))

        # Commit flip when candidate has sufficient confirmations
        if getattr(self.counters, candidate.value) >= hysteresis_hits and candidate != self.current:
            self.current = candidate
            self.last_flip_ts = ts_bar
            self.since_ts = ts_bar
            self.cooldown_left = cooldown_bars
            self.cooldown_total = cooldown_bars
        elif candidate == self.current:
            if self.since_ts == 0:
                self.since_ts = ts_bar
            if self.cooldown_left > 0:
                self.cooldown_left -= 1
        return self.current

def _bucket_vol(atr_pct: Optional[float], low_thr: float, high_thr: float) -> VolatilityBucket:
    if atr_pct is None:
        return VolatilityBucket.MEDIUM
    if atr_pct < low_thr:
        return VolatilityBucket.LOW
    if atr_pct > high_thr:
        return VolatilityBucket.HIGH
    return VolatilityBucket.MEDIUM

def _vol_state(atr_pct: Optional[float], crush_thr: float, expansion_thr: float) -> VolState:
    if atr_pct is None:
        return VolState.NORMAL
    if atr_pct <= crush_thr:
        return VolState.CRUSH
    if atr_pct >= expansion_thr:
        return VolState.EXPANSION
    return VolState.NORMAL

def _primary_candidate(
    adx: Optional[float],
    atr_pct: Optional[float],
    ema_sep_atr: Optional[float],
    bb_width_pct_atr: Optional[float],
    cfg: Dict,
    current: RegimeMode,
) -> RegimeMode:
    """Return next regime candidate honoring enter/exit hysteresis.

    Logic:
      1. Compute raw enters for each regime (independent rules).
      2. If current regime still within its *exit* thresholds -> stay.
      3. Else pick highest-priority satisfied enter (trend > mean_revert > chop) else fallback to chop.
    """
    if adx is None:
        return current
    prim = cfg.get("primary", {})
    # Threshold dicts
    t_enter = prim.get("trend", {}).get("enter", {})
    t_exit = prim.get("trend", {}).get("exit", {})
    mr_enter = prim.get("mean_revert", {}).get("enter", {})
    mr_exit = prim.get("mean_revert", {}).get("exit", {})
    c_enter = prim.get("chop", {}).get("enter", {})
    c_exit = prim.get("chop", {}).get("exit", {})

    # Helper lambdas
    def _trend_enter() -> bool:
        if adx < t_enter.get("adx_min", 24):
            return False
        if ema_sep_atr is not None and ema_sep_atr < t_enter.get("ema_sep_atr_min", 0.35):
            return False
        return True

    def _trend_hold() -> bool:
        # Stay in trend unless BOTH core metrics fall below exit thresholds
        if adx >= t_exit.get("adx_min", t_enter.get("adx_min", 24) - 6):
            return True
        if ema_sep_atr is not None and ema_sep_atr >= t_exit.get("ema_sep_atr_min", t_enter.get("ema_sep_atr_min", 0.35) - 0.15):
            return True
        return False

    def _mr_enter() -> bool:
        if adx > mr_enter.get("adx_max", 16):
            return False
        if bb_width_pct_atr is not None and bb_width_pct_atr > mr_enter.get("bb_width_pct_atr_max", 0.70):
            return False
        return True

    def _mr_hold() -> bool:
        if adx <= mr_exit.get("adx_max", mr_enter.get("adx_max", 16) + 4):
            return True
        return False

    def _chop_enter() -> bool:
        return adx <= c_enter.get("adx_max", 20)

    def _chop_hold() -> bool:
        return adx <= c_exit.get("adx_max", c_enter.get("adx_max", 20) + 4)

    # Hold logic first
    if current == RegimeMode.TREND and _trend_hold():
        return RegimeMode.TREND
    if current == RegimeMode.MEAN_REVERT and _mr_hold():
        return RegimeMode.MEAN_REVERT
    if current == RegimeMode.CHOP and _chop_hold():
        return RegimeMode.CHOP

    # Evaluate fresh enters (priority order)
    if _trend_enter():
        return RegimeMode.TREND
    if _mr_enter():
        return RegimeMode.MEAN_REVERT
    if _chop_enter():
        return RegimeMode.CHOP
    return RegimeMode.CHOP

def _strong_override_flag(adx: Optional[float], ema_sep_atr: Optional[float], cfg: Dict) -> bool:
    """Return True if an exceptionally strong trend should override cooldown."""
    if adx is None:
        return False
    prim = cfg.get("primary", {}).get("trend", {}).get("enter", {})
    enter_adx = prim.get("adx_min", 24)
    enter_sep = prim.get("ema_sep_atr_min", 0.35)
    if adx >= enter_adx + 6 and (ema_sep_atr is None or ema_sep_atr >= enter_sep + 0.10):
        return True
    return False

def _liquidity_state(spread_bps: Optional[float], vol_z: Optional[float], cfg: Dict) -> LiquidityState:
    liq_cfg = cfg.get("liquidity", {})
    max_spread = float(liq_cfg.get("max_spread_bp", 3.0))
    min_vol_z = float(liq_cfg.get("min_volume_z", -1.0))
    if spread_bps is not None and spread_bps > max_spread:
        return LiquidityState.THIN
    if vol_z is not None and vol_z < min_vol_z:
        return LiquidityState.THIN
    return LiquidityState.OK

def _confidence(adx: Optional[float], atr_pct: Optional[float], ema_sep_atr: Optional[float]) -> float:
    parts = []
    if adx is not None: parts.append(min(1.0, adx / 50.0))
    if atr_pct is not None: parts.append(max(0.0, min(1.0, atr_pct)))
    if ema_sep_atr is not None: parts.append(min(1.0, ema_sep_atr / 0.6))
    if not parts:
        return 0.0
    return float(sum(parts) / len(parts))

def classify_regime_full(
    adx: Optional[float],
    atr_pct: Optional[float],
    ema_sep_atr: Optional[float],
    settings: Dict,
    state: RegimeStateMachine,
    *,
    bb_width_pct_atr: Optional[float] = None,
    volume_z: Optional[float] = None,
    spread_bps: Optional[float] = None,
    ts_bar: Optional[int] = None,
    news_flag: Optional[bool] = None,
) -> RegimeFeatures:
    cfg = (settings.get("regime") or {})
    hysteresis_hits = int(cfg.get("hysteresis_hits", 2))
    cooldown_bars = int(cfg.get("cooldown_bars", 8))
    allow_override = bool(cfg.get("strong_override", True))
    # Determine candidate honoring exit thresholds
    candidate = _primary_candidate(adx, atr_pct, ema_sep_atr, bb_width_pct_atr, cfg, state.current)
    strong_flag = _strong_override_flag(adx, ema_sep_atr, cfg)
    now_ts = ts_bar if ts_bar is not None else int(time.time())
    committed = state.update(candidate, hysteresis_hits, cooldown_bars, now_ts, allow_override, strong_flag)
    vol_bucket = _bucket_vol(atr_pct, cfg.get("atr_low_thr", 0.35), cfg.get("atr_high_thr", 0.65))
    vol_cfg = cfg.get("vol", {}) or {}
    v_state = _vol_state(atr_pct, vol_cfg.get("crush_atr_pct", 0.20), vol_cfg.get("expansion_atr_pct", 0.70))
    n_state = NewsState.NEWS if news_flag else NewsState.QUIET
    profile = RegimeProfile(committed.value)
    liq_state = _liquidity_state(spread_bps, volume_z, cfg)
    conf = _confidence(adx, atr_pct, ema_sep_atr)
    gates = {
        "trend_following": committed == RegimeMode.TREND and liq_state == LiquidityState.OK,
        "mean_reversion": committed in (RegimeMode.MEAN_REVERT, RegimeMode.CHOP) and v_state != VolState.EXPANSION,
        "breakout": committed == RegimeMode.TREND and v_state != VolState.CRUSH
    }
    return RegimeFeatures(
        adx=adx,
        atr_percentile=atr_pct,
        vol_bucket=vol_bucket,
        mode=committed,
        profile=profile,
        vol_state=v_state,
        news_state=n_state,
        gates=gates,
        liquidity=liq_state,
        confidence=conf,
        since_ts=state.since_ts or now_ts,
        last_flip_ts=state.last_flip_ts or now_ts,
    )

_GLOBAL_STATE = RegimeStateMachine()

def classify_regime(adx: Optional[float], atr_pct: Optional[float], *_args, **_kwargs) -> Dict:
    rf = classify_regime_full(adx, atr_pct, None, {"regime": {}}, _GLOBAL_STATE)
    return {"mode": rf.mode.value, "vol_bucket": rf.vol_bucket.value, "profile": rf.profile.value}
