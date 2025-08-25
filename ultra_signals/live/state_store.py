"""Persistent runtime state (SQLite) for live trading.

Tables (created lazily, WAL mode):
  positions(symbol TEXT PRIMARY KEY, qty REAL, avg_px REAL, ts INTEGER)
  orders_outbox(client_order_id TEXT PRIMARY KEY, status TEXT, exchange_order_id TEXT,
                last_error TEXT, retries INTEGER DEFAULT 0, ts INTEGER)
  risk_runtime(key TEXT PRIMARY KEY, value TEXT)
  offsets(topic TEXT, symbol TEXT, timeframe TEXT, last_ts INTEGER,
          PRIMARY KEY(topic, symbol, timeframe))

Only minimal fields needed now; schema can be extended without breaking code
by adding new columns with defaults.
"""
from __future__ import annotations
import sqlite3
import threading
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, Tuple
from loguru import logger

_DDL = [
    "PRAGMA journal_mode=WAL;",
    "CREATE TABLE IF NOT EXISTS positions (symbol TEXT PRIMARY KEY, qty REAL, avg_px REAL, ts INTEGER)",
    "CREATE TABLE IF NOT EXISTS orders_outbox (client_order_id TEXT PRIMARY KEY, status TEXT, exchange_order_id TEXT, last_error TEXT, retries INTEGER DEFAULT 0, ts INTEGER, filled_qty REAL)",
    "CREATE TABLE IF NOT EXISTS risk_runtime (key TEXT PRIMARY KEY, value TEXT)",
    "CREATE TABLE IF NOT EXISTS offsets (topic TEXT, symbol TEXT, timeframe TEXT, last_ts INTEGER, PRIMARY KEY(topic, symbol, timeframe))",
]

class StateStore:
    def __init__(self, path: str = "live_state.db"):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):  # pragma: no cover (idempotent setup)
        cur = self._conn.cursor()
        for stmt in _DDL:
            try:
                cur.execute(stmt)
            except Exception as e:
                logger.error(f"StateStore DDL error: {e} :: {stmt}")
        # Backwards-compatible migration: ensure filled_qty column exists
        try:
            cur.execute("ALTER TABLE orders_outbox ADD COLUMN filled_qty REAL")
        except Exception:
            pass
        self._conn.commit()

    # ---------------- Generic helpers ----------------
    def _execute(self, sql: str, params: Iterable[Any] = ()):  # small wrapper
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            self._conn.commit()
            return cur

    # ---------------- Positions -----------------------
    def upsert_position(self, symbol: str, qty: float, avg_px: float, ts: Optional[int] = None):
        ts = ts or int(time.time() * 1000)
        self._execute(
            "INSERT INTO positions(symbol, qty, avg_px, ts) VALUES(?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty, avg_px=excluded.avg_px, ts=excluded.ts",
            (symbol, qty, avg_px, ts),
        )

    def get_positions(self) -> Dict[str, Dict[str, float]]:
        cur = self._execute("SELECT * FROM positions")
        return {r["symbol"]: {"qty": r["qty"], "avg_px": r["avg_px"], "ts": r["ts"]} for r in cur.fetchall()}

    # ---------------- Orders Outbox -------------------
    def ensure_order(self, client_order_id: str) -> bool:
        """Register an outbound order idempotently.
        Returns True if newly inserted; False if already exists.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT 1 FROM orders_outbox WHERE client_order_id=?", (client_order_id,))
            if cur.fetchone():
                return False
            cur.execute(
                "INSERT INTO orders_outbox(client_order_id, status, ts) VALUES(?,?,?)",
                (client_order_id, "PENDING", int(time.time() * 1000)),
            )
            self._conn.commit()
            return True

    def update_order(self, client_order_id: str, **fields):
        if not fields:
            return
        cols = ",".join(f"{k}=?" for k in fields.keys())
        params = list(fields.values()) + [client_order_id]
        self._execute(f"UPDATE orders_outbox SET {cols} WHERE client_order_id=?", params)

    def get_order(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        cur = self._execute("SELECT * FROM orders_outbox WHERE client_order_id=?", (client_order_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_orders(self, status: Optional[str] = None) -> list[Dict[str, Any]]:
        if status:
            cur = self._execute("SELECT * FROM orders_outbox WHERE status=?", (status,))
        else:
            cur = self._execute("SELECT * FROM orders_outbox")
        return [dict(r) for r in cur.fetchall()]

    # ---------------- Risk runtime --------------------
    def set_risk_value(self, key: str, value: Any):
        self._execute(
            "INSERT INTO risk_runtime(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    def get_risk_value(self, key: str, default=None):
        cur = self._execute("SELECT value FROM risk_runtime WHERE key=?", (key,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else default

    # ---------------- Offsets ------------------------
    def set_offset(self, topic: str, symbol: str, timeframe: str, last_ts: int):
        self._execute(
            "INSERT INTO offsets(topic,symbol,timeframe,last_ts) VALUES(?,?,?,?) ON CONFLICT(topic,symbol,timeframe) DO UPDATE SET last_ts=excluded.last_ts",
            (topic, symbol, timeframe, last_ts),
        )

    def get_offset(self, topic: str, symbol: str, timeframe: str) -> Optional[int]:
        cur = self._execute(
            "SELECT last_ts FROM offsets WHERE topic=? AND symbol=? AND timeframe=?",
            (topic, symbol, timeframe),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None

    # ---------------- Housekeeping -------------------
    def close(self):  # pragma: no cover
        try:
            self._conn.close()
        except Exception:
            pass

__all__ = ["StateStore"]
