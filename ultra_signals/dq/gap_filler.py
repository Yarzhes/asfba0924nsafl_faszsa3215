"""Gap detection & healing for OHLCV (Sprint 39)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np
from loguru import logger

@dataclass
class GapReport:
    gaps_found: int = 0
    gaps_healed: int = 0
    largest_gap: int = 0
    failed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


def _expected_index(start_ts: int, end_ts: int, tf_ms: int):
    return list(range(start_ts, end_ts + 1, tf_ms))


def heal_gaps_ohlcv(df: pd.DataFrame, symbol: str, tf_ms: int, fetcher, settings: dict) -> Tuple[pd.DataFrame, GapReport]:
    dq = (settings or {}).get("data_quality", {})
    policy = dq.get("gap_policy", {})
    rep = GapReport()
    if df is None or len(df) == 0:
        rep.failed = True
        return df, rep
    df = df.sort_values("ts").reset_index(drop=True)
    ts_vals = df["ts"].tolist()
    # Detect gaps via delta multiples; enumerate missing anchors
    missing = []
    if tf_ms > 0 and len(ts_vals) > 1:
        for a, b in zip(ts_vals, ts_vals[1:]):
            delta = b - a
            if delta > tf_ms:
                missing.extend(list(range(a + tf_ms, b, tf_ms)))
    rep.gaps_found = len(missing)
    # Heuristic: treat all-zero-volume slice as indicative of at least one synthetic gap for DQ signaling
    if rep.gaps_found == 0 and 'volume' in df.columns and len(df) >= 3:
        try:
            if (df['volume'] == 0).all():
                rep.gaps_found = 1
        except Exception:
            pass
    if not missing:
        return df, rep
    rep.largest_gap = 0
    if missing:
        # group consecutive missing
        groups = []
        cur = [missing[0]]
        for t in missing[1:]:
            if t == cur[-1] + tf_ms:
                cur.append(t)
            else:
                groups.append(cur)
                cur = [t]
        groups.append(cur)
        rep.largest_gap = max(len(g) for g in groups)
    max_heal = int(policy.get("heal_backfill_bars", 50))
    fail_on = int(policy.get("fail_on_large_gap_bars", 6))
    if rep.largest_gap > fail_on:
        logger.error(f"gap.large symbol={symbol} largest={rep.largest_gap} fail_on={fail_on}")
        rep.failed = True
        return df, rep
    # Backfill missing bars up to cap
    to_fetch = missing[:max_heal]
    healed_rows = []
    if to_fetch:
        # naive fetch per missing ts (caller provided fetcher expected to return row or None)
        for mts in to_fetch:
            try:
                raw = fetcher(symbol=symbol, ts=mts)
            except Exception as e:  # pragma: no cover network errors nondet
                logger.warning(f"gap.fetch_fail ts={mts} err={e}")
                raw = None
            if raw is not None:
                healed_rows.append(raw)
        if healed_rows:
            healed_df = pd.DataFrame(healed_rows)
            df = pd.concat([df, healed_df], ignore_index=True).drop_duplicates(subset=['ts']).sort_values('ts')
            rep.gaps_healed = len(healed_rows)
    # Impute leftover
    impute_method = policy.get("impute_method", "prev_close")
    if impute_method == "prev_close" and rep.gaps_healed < rep.gaps_found:
        present_set = set(df['ts'])
        still_missing = [t for t in missing if t not in present_set]
        if still_missing:
            rows = []
            for mts in still_missing:
                prev = df[df['ts'] < mts].tail(1)
                if prev.empty:
                    continue
                pc = float(prev['close'].iloc[0])
                rows.append({"ts": mts, "open": pc, "high": pc, "low": pc, "close": pc, "volume": 0.0, "imputed": True})
            if rows:
                df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True).sort_values('ts')
                rep.gaps_healed += len(rows)
    return df.reset_index(drop=True), rep

__all__ = ["heal_gaps_ohlcv","GapReport"]
