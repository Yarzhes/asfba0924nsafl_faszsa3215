"""Data validation utilities for ticks & OHLCV (Sprint 39)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

@dataclass
class ValidationReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str):  # pragma: no cover trivial
        self.warnings.append(msg)

_DEF_COLS = ["ts","open","high","low","close","volume"]


def _enforce_numeric(df: pd.DataFrame, strict: bool, report: ValidationReport):
    for col in [c for c in df.columns if c != "ts"]:
        if not np.isfinite(df[col]).all():
            msg = f"non_finite_values col={col}"
            if strict:
                report.add_error(msg)
            else:  # pragma: no cover
                report.add_warning(msg)
        # replace inf with nan then drop later
        df[col] = pd.to_numeric(df[col], errors='coerce')


def _clamp_decimals(df: pd.DataFrame, price_max: int, vol_max: int):
    q_price = 10 ** price_max
    q_vol = 10 ** vol_max
    for col in ["open","high","low","close"]:
        if col in df:
            df[col] = (df[col] * q_price).round().div(q_price)
    if "volume" in df:
        df["volume"] = (df["volume"] * q_vol).round().div(q_vol)


def validate_ohlcv_df(df: pd.DataFrame, tf_ms: int, settings: dict, symbol: str, venue: str) -> ValidationReport:
    dq = (settings or {}).get("data_quality", {})
    rep = ValidationReport(ok=True)
    if df is None or len(df) == 0:
        rep.add_error("empty_df")
        return rep
    cols = dq.get("ohlcv_schema", _DEF_COLS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        rep.add_error(f"missing_cols={missing}")
        return rep
    # Detect out-of-order BEFORE sorting so test harness sees a warning
    orig_ts = df["ts"].values
    if (np.diff(orig_ts) < 0).any():
        deltas_o = np.diff(orig_ts)
        out_of_order = int((deltas_o < 0).sum())
        max_back = int(abs(deltas_o[deltas_o < 0].min())) if (deltas_o < 0).any() else 0
        tolerance = dq.get("max_out_of_order_ms", 2000)
        if max_back > tolerance:
            rep.add_error(f"out_of_order max_back={max_back}ms tolerance={tolerance}")
        else:
            rep.add_warning(f"minor_out_of_order count={out_of_order} max_back={max_back}")
    # Work with sorted copy downstream
    df = df.sort_values("ts")
    ts = df["ts"].values
    # Duplicates
    if dq.get("drop_duplicates", True):
        before = len(df)
        df = df.drop_duplicates(subset=["ts"], keep="last")
        dropped = before - len(df)
        if dropped:
            rep.stats["duplicates_dropped"] = dropped
    elif not dq.get("allow_equal_timestamps", False) and df["ts"].duplicated().any():
        rep.add_error("duplicate_timestamps")
    # Numeric & decimals
    _enforce_numeric(df, dq.get("strict_numeric", True), rep)
    _clamp_decimals(df, dq.get("price_decimals_max", 8), dq.get("volume_decimals_max", 8))
    # Coverage check
    expected = ts[-1] - ts[0]
    if expected > 0 and tf_ms > 0:
        bars = int(round(expected / tf_ms)) + 1
        coverage = (len(df) / max(1, bars)) * 100.0
        rep.stats["coverage_pct"] = coverage
        min_cov = dq.get("gap_policy", {}).get("min_bar_coverage_pct", 95)
        if coverage < min_cov:
            rep.add_warning(f"low_coverage coverage_pct={coverage:.2f} min={min_cov}")
    rep.stats["rows"] = len(df)
    rep.stats["symbol"] = symbol
    rep.stats["venue"] = venue
    rep.stats["tf_ms"] = tf_ms
    return rep


def validate_tick_df(df: pd.DataFrame, settings: dict, symbol: str, venue: str) -> ValidationReport:
    dq = (settings or {}).get("data_quality", {})
    rep = ValidationReport(ok=True)
    if df is None or len(df) == 0:
        rep.add_error("empty_df")
        return rep
    if "ts" not in df.columns:
        rep.add_error("missing_ts")
        return rep
    df = df.sort_values("ts")
    # duplicates
    if dq.get("drop_duplicates", True):
        before = len(df)
        df = df.drop_duplicates(subset=["ts"], keep="last")
        dropped = before - len(df)
        if dropped:
            rep.stats["duplicates_dropped"] = dropped
    elif not dq.get("allow_equal_timestamps", False) and df["ts"].duplicated().any():
        rep.add_error("duplicate_timestamps")
    _enforce_numeric(df, dq.get("strict_numeric", True), rep)
    rep.stats["rows"] = len(df)
    rep.stats["symbol"] = symbol
    rep.stats["venue"] = venue
    return rep

__all__ = ["validate_tick_df","validate_ohlcv_df","ValidationReport"]
