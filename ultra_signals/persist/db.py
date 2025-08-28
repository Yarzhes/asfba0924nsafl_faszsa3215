"""SQLite persistence layer (Sprint 25).

Provides a single access point `get_db()` for reuse across modules. Uses WAL mode
and versioned migrations tracked in `migrations` table. Designed for exactly-once
order journaling + recovery.

Schema (initial): see migrations list in `migrations.py`.
"""
from __future__ import annotations
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Any, Dict
from loguru import logger

_DB_PATH_ENV = "ULTRA_SIGNALS_DB_PATH"
_DEFAULT_PATH = "live_state.db"  # reuse existing name; now managed centrally

_lock = threading.RLock()
_conn: Optional[sqlite3.Connection] = None

PRAGMAS = [
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    "PRAGMA foreign_keys=ON;",
]

@contextmanager
def tx(immediate: bool = False) -> Iterator[sqlite3.Cursor]:
    """Context manager for a DB transaction.

    Args:
        immediate: If True issues BEGIN IMMEDIATE for write-intent early lock
                    acquisition (used for journal-first order write pattern).
    """
    cur = None
    try:
        with _lock:
            if _conn is None:
                raise RuntimeError("DB not initialised. Call init_db() first.")
            if immediate:
                _conn.execute("BEGIN IMMEDIATE")
            else:
                _conn.execute("BEGIN")
            cur = _conn.cursor()
        yield cur
        with _lock:
            _conn.commit()
    except Exception:
        if _conn:
            with _lock:
                _conn.rollback()
        raise
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:  # pragma: no cover
                pass

def init_db(path: str | None = None):
    """Initialise global connection + apply pragmas (idempotent)."""
    global _conn
    with _lock:
        if _conn is not None:
            return
        db_path = Path(path or os.environ.get(_DB_PATH_ENV) or _DEFAULT_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        _conn.row_factory = sqlite3.Row
        for p in PRAGMAS:
            try:
                _conn.execute(p)
            except Exception as e:  # pragma: no cover
                logger.warning(f"[DB] pragma failed {p} :: {e}")
        logger.info(f"[DB] Opened {db_path} (WAL mode)")


def get_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("DB not initialised; call init_db()")
    return _conn


def fetchall(sql: str, params: tuple[Any, ...] = ()):
    with _lock:
        cur = get_conn().execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
    return rows

def fetchone(sql: str, params: tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
    with _lock:
        cur = get_conn().execute(sql, params)
        row = cur.fetchone()
        cur.close()
    return dict(row) if row else None

def execute(sql: str, params: tuple[Any, ...] = ()):  # autocommit helper
    with tx() as cur:
        cur.execute(sql, params)


# Sprint 29: record liquidity gate decision
def record_liquidity_decision(symbol: str, ts: int, profile: str, action: str, reason: str | None, meta: Dict[str, Any] | None):
    try:
        spread = meta.get('spread_bps') if meta else None
        impact = meta.get('impact_50k') if meta else None
        dr = meta.get('dr') if meta else None
        rv = meta.get('rv_5s') if meta else None
        source = meta.get('source') if meta else None
    except Exception:
        spread=impact=dr=rv=source=None
    try:
        with tx() as cur:
            cur.execute(
                """INSERT INTO liquidity_decisions(ts,symbol,profile,action,reason,spread_bps,impact_50k,dr,rv_5s,source)
                      VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (ts, symbol, profile, action, reason, spread, impact, dr, rv, source)
            )
    except Exception as e:  # pragma: no cover
        logger.debug(f"[DB] liquidity decision insert failed: {e}")


# Record policy action decisions (drift / retrain / pause / shrink)
def record_policy_action(symbol: str, ts: int, action_type: str, size_mult: float | None, reason_codes: str | None, meta: Dict[str, Any] | None = None):
    """Persist a compact policy action for audit and dashboarding.

    This follows the same best-effort pattern as other record helpers.
    """
    try:
        # normalize
        rc = ','.join(reason_codes) if isinstance(reason_codes, (list, tuple)) else (reason_codes or None)
    except Exception:
        rc = None
    try:
        with tx() as cur:
            cur.execute(
                """INSERT INTO policy_actions(ts,symbol,action_type,size_mult,reason_codes,meta)
                      VALUES(?,?,?,?,?,?)""",
                (ts, symbol, action_type, size_mult, rc, str(meta) if meta is not None else None)
            )
    except Exception as e:  # pragma: no cover
        logger.debug(f"[DB] policy action insert failed: {e}")


def write_retrain_job(queue_dir: str, job: Dict[str, Any]):
    """Durably write a retrain job as one-line JSONL into queue_dir.

    This is a simple fallback delivery mechanism until S27 API details are provided.
    """
    try:
        import json
        p = Path(queue_dir)
        p.mkdir(parents=True, exist_ok=True)
        fname = f"retrain_{int(time.time()*1000)}_{os.getpid()}.jsonl"
        full = p / fname
        with full.open('w', encoding='utf-8') as f:
            f.write(json.dumps(job, default=str) + "\n")
    except Exception as e:
        logger.debug(f"[DB] write_retrain_job failed: {e}")


def upsert_order_pending(order: Dict[str, Any]):
    """Journal-first write: insert PENDING order if absent.
    order dict MUST include: client_order_id, venue, symbol, side, type, qty, price, reduce_only,
    parent_id, profile_id, cfg_hash.
    """
    now = int(time.time() * 1000)
    with tx(immediate=True) as cur:
        cur.execute(
            """
            INSERT OR IGNORE INTO orders_outbox(
              client_order_id, venue, symbol, side, type, qty, price, reduce_only,
              parent_id, status, venue_order_id, last_error, retries, created_ts,
              updated_ts, profile_id, cfg_hash
            ) VALUES(?,?,?,?,?,?,?,?,?,'PENDING',NULL,NULL,0,?, ?,?,?)
            """,
            (
                order.get("client_order_id"), order.get("venue"), order.get("symbol"), order.get("side"),
                order.get("type"), order.get("qty"), order.get("price"), int(order.get("reduce_only", False)),
                order.get("parent_id"), now, now, order.get("profile_id"), order.get("cfg_hash"),
            ),
        )


def update_order_after_ack(client_order_id: str, *, status: str, venue_order_id: str | None):
    now = int(time.time() * 1000)
    with tx() as cur:
        cur.execute(
            "UPDATE orders_outbox SET status=?, venue_order_id=?, updated_ts=? WHERE client_order_id=?",
            (status, venue_order_id, now, client_order_id),
        )


def record_fill(fill: Dict[str, Any]):
    """Insert fill (idempotent) and update position atomically."""
    now = int(time.time() * 1000)
    with tx() as cur:
        cur.execute(
            """INSERT OR IGNORE INTO exec_fills(fill_id, client_order_id, venue, venue_order_id, symbol, qty, price, fee, is_maker, ts)
                 VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                fill.get("fill_id"), fill.get("client_order_id"), fill.get("venue"), fill.get("venue_order_id"),
                fill.get("symbol"), fill.get("qty"), fill.get("price"), fill.get("fee"), int(fill.get("is_maker",0)), fill.get("ts", now)
            ),
        )
        # Recompute position for symbol from all fills (append-only correctness)
        cur.execute("SELECT symbol, qty, price FROM exec_fills WHERE symbol=?", (fill.get("symbol"),))
        rows = cur.fetchall()
        qty = sum(r[1] for r in rows)
        avg_px = 0.0
        if rows:
            gross = sum(r[1]*r[2] for r in rows)
            total_abs = sum(abs(r[1]) for r in rows) or 1.0
            avg_px = gross / total_abs
        cur.execute(
            "INSERT INTO positions(symbol, qty, avg_px, realized_pnl, updated_ts, venue, hedge) VALUES(?,?,?,?,?,?,?)\n             ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty, avg_px=excluded.avg_px, updated_ts=excluded.updated_ts",
            (
                fill.get("symbol"), qty, avg_px, 0.0, now, fill.get("venue"), 0,
            ),
        )


def upsert_offset(stream: str, last_ts: int, last_seq: int | None = None):
    with tx() as cur:
        cur.execute(
            "INSERT INTO offsets(stream,last_ts,last_seq) VALUES(?,?,?) ON CONFLICT(stream) DO UPDATE SET last_ts=excluded.last_ts, last_seq=excluded.last_seq",
            (stream, last_ts, last_seq),
        )


def get_offset(stream: str) -> Optional[Dict[str, Any]]:
    return fetchone("SELECT * FROM offsets WHERE stream=?", (stream,))


def snapshot_settings_fingerprint(cfg_hash: str, profile_version: str | None):
    with tx() as cur:
        cur.execute(
            "INSERT INTO settings_fingerprint(id,cfg_hash,profile_version) VALUES(1,?,?) ON CONFLICT(id) DO UPDATE SET cfg_hash=excluded.cfg_hash, profile_version=excluded.profile_version",
            (cfg_hash, profile_version),
        )

def upsert_oco_link(parent_client_id: str, stop_client_id: str | None, tp_client_ids: list[str] | None):
    tp_str = ",".join(tp_client_ids) if tp_client_ids else None
    with tx() as cur:
        cur.execute(
            "INSERT INTO oco_links(parent_client_id, stop_client_id, tp_client_ids) VALUES(?,?,?) ON CONFLICT(parent_client_id) DO UPDATE SET stop_client_id=excluded.stop_client_id, tp_client_ids=excluded.tp_client_ids",
            (parent_client_id, stop_client_id, tp_str)
        )

def fetch_oco_link(parent_client_id: str):
    row = fetchone("SELECT * FROM oco_links WHERE parent_client_id=?", (parent_client_id,))
    if not row:
        return None
    tps = row.get('tp_client_ids')
    if tps:
        row['tp_client_ids'] = [t for t in tps.split(',') if t]
    else:
        row['tp_client_ids'] = []
    return row

__all__ = [
    'init_db','get_conn','tx','upsert_order_pending','update_order_after_ack','record_fill',
    'upsert_offset','get_offset','snapshot_settings_fingerprint','fetchall','fetchone','execute',
    'upsert_oco_link','fetch_oco_link'
]
