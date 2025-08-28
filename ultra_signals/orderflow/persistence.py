"""FeatureView persistence for orderflow micro-features.

This module provides a small, import-safe sqlite-backed writer used in
tests and lightweight runtimes. It supports schema migration (adds new
columns when discovered missing) so that existing DB files remain usable.

The writer stores both flattened metric columns for fast querying and a
`components` JSON payload for richer, versioned data.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import time
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class OrderflowConfig:
    """Tunable knobs for the orderflow engine (kept here for convenience).

    These values are defaults only — actual signal code should load
    configuration from the project's config system.
    """

    depth_levels: List[int] = dataclasses.field(default_factory=lambda: [1, 5, 50])
    cvd_window_s: int = 60
    vps_window_s: int = 10
    burst_sigma: float = 2.0
    min_footprint_volume: float = 1000.0
    include_liquidations: bool = False


class FeatureViewWriter:
    """Sqlite-backed writer for orderflow FeatureView.

    Guarantees:
      - Creates `orderflow_features` table if missing
      - Adds new columns if they are not present (attempts safe migration)
      - Stores a JSON `components` blob for extensible fields
    """

    BASE_COLUMNS = {
        "ts": "INTEGER",
        "symbol": "TEXT",
        "of_micro_score": "REAL",
        "components": "TEXT",
        "price": "REAL",
    }

    EXTRA_COLUMNS = {
        # CVD
        "cvd_abs": "REAL",
        "cvd_pct": "REAL",
        "cvd_z": "REAL",
        "cvd_div_flag": "INTEGER",
        # Order book imbalance
        "ob_imbalance_top1": "REAL",
        "ob_imbalance_top5": "REAL",
        "ob_imbalance_full": "REAL",
        # Tape metrics
        "tape_tps": "REAL",
        "tape_vps": "REAL",
        "tape_nps": "REAL",
        "tape_burst_flag": "INTEGER",
        # Footprint
        "footprint_sr_level_px": "REAL",
        "footprint_sr_strength": "REAL",
        "footprint_absorption_flag": "INTEGER",
    # VWAP / execution slice metrics
    "expected_cost_bps": "REAL",
    "realized_cost_bps": "REAL",
    "schedule_lag": "REAL",
    # per-slice identifier for VWAP/TWAP persistence and post-fill linkage
    "slice_id": "TEXT",
    # TCA persisted aggregates (per-slice or per-venue derived)
    "tca_slip_bps": "REAL",
    "tca_fill_ratio": "REAL",
    "tca_latency_ms": "REAL",
    # ranking & versioning for feature view models
    "tca_venue_rank": "TEXT",
    "tca_version": "TEXT",
    }

    def __init__(self, sqlite_path: str = "orderflow_features.db", config: Optional[OrderflowConfig] = None) -> None:
        self._conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._config = config or OrderflowConfig()
        cur = self._conn.cursor()
        # create metadata table
        cur.execute(
            "CREATE TABLE IF NOT EXISTS featureview_meta (k TEXT PRIMARY KEY, v TEXT)"
        )
        # create base table if missing
        cols = ", ".join(f"{k} {t}" for k, t in self.BASE_COLUMNS.items())
        cur.execute(f"CREATE TABLE IF NOT EXISTS orderflow_features ({cols})")
        self._conn.commit()
        # ensure extra columns exist
        self._ensure_columns()

    def _existing_columns(self) -> List[str]:
        cur = self._conn.cursor()
        cur.execute("PRAGMA table_info(orderflow_features)")
        return [r[1] for r in cur.fetchall()]

    def _ensure_columns(self) -> None:
        existing = set(self._existing_columns())
        cur = self._conn.cursor()
        for col, typ in self.EXTRA_COLUMNS.items():
            if col not in existing:
                try:
                    cur.execute(f"ALTER TABLE orderflow_features ADD COLUMN {col} {typ}")
                except Exception:
                    # best-effort: if ALTER fails for any reason, continue.
                    # This keeps the module import-safe during tests.
                    pass
        self._conn.commit()

    def set_meta(self, key: str, value: Any) -> None:
        cur = self._conn.cursor()
        cur.execute("INSERT OR REPLACE INTO featureview_meta (k,v) VALUES (?,?)", (key, json.dumps(value)))
        self._conn.commit()

    def get_meta(self, key: str) -> Optional[Any]:
        cur = self._conn.cursor()
        cur.execute("SELECT v FROM featureview_meta WHERE k=?", (key,))
        r = cur.fetchone()
        if not r:
            return None
        return json.loads(r[0])

    def write_record(self, record: Dict[str, Any]) -> None:
        """Write a single orderflow record.

        Accepts both flattened keys (e.g. `cvd_abs`) and a `components` dict
        which will be stored verbatim as JSON. Timestamps default to now.
        """

        cur = self._conn.cursor()
        comps = json.dumps(record.get("components") or {})
        ts = int(record.get("ts") or int(time.time()))
        symbol = str(record.get("symbol") or "")

        # collect column names and values dynamically to be forward-compatible
        cols = ["ts", "symbol", "of_micro_score", "components", "price"]
        vals: List[Any] = [
            ts,
            symbol,
            None if record.get("of_micro_score") is None else float(record.get("of_micro_score")),
            comps,
            None if record.get("price") is None else float(record.get("price")),
        ]

        for k in self.EXTRA_COLUMNS.keys():
            cols.append(k)
            v = record.get(k)
            if v is None:
                vals.append(None)
            else:
                # flags expected as bool/int; coerce to int for INTEGER columns
                if isinstance(v, bool):
                    vals.append(1 if v else 0)
                else:
                    vals.append(v)

        q = f"INSERT INTO orderflow_features ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
        cur.execute(q, tuple(vals))
        self._conn.commit()

    def update_by_slice_id(self, slice_id: str, updates: Dict[str, Any]) -> None:
        """Update flattened columns for a record matching slice_id."""
        if not slice_id or not updates:
            return
        cur = self._conn.cursor()
        # build SET clause
        set_parts = []
        vals = []
        for k, v in updates.items():
            if k not in self.BASE_COLUMNS and k not in self.EXTRA_COLUMNS:
                continue
            set_parts.append(f"{k} = ?")
            # coerce booleans
            if isinstance(v, bool):
                vals.append(1 if v else 0)
            else:
                vals.append(v)
        if not set_parts:
            return
        vals.append(slice_id)
        q = f"UPDATE orderflow_features SET {', '.join(set_parts)} WHERE slice_id = ?"
        try:
            cur.execute(q, tuple(vals))
            self._conn.commit()
        except Exception:
            pass

    def query_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM orderflow_features ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            data: Dict[str, Any] = {k: r[k] for k in r.keys()}
            # parse components JSON
            try:
                data["components"] = json.loads(data.get("components") or "{}")
            except Exception:
                data["components"] = {}
            out.append(data)
        return out

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


__all__ = ["FeatureViewWriter", "OrderflowConfig"]
