"""SQLite persistence helpers for events table."""
from __future__ import annotations
from typing import Iterable, Dict, Any, List
from loguru import logger
from ultra_signals.persist.db import tx, fetchall


DDL_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  provider TEXT,
  name TEXT,
  category TEXT,
  country TEXT NULL,
  symbol_scope TEXT,
  importance INTEGER,
  start_ts INTEGER,
  end_ts INTEGER,
  source_payload TEXT,
  inserted_ts INTEGER
);
CREATE INDEX IF NOT EXISTS ix_events_time ON events(start_ts, end_ts);
"""


def upsert_events(rows: Iterable[Dict[str, Any]]):
    rows = list(rows)
    if not rows:
        return 0
    inserted = 0
    with tx() as cur:
        for r in rows:
            cur.execute(
                """INSERT OR REPLACE INTO events(id,provider,name,category,country,symbol_scope,importance,start_ts,end_ts,source_payload,inserted_ts)
                       VALUES(?,?,?,?,?,?,?,?,?,?,strftime('%s','now')*1000)""",
                (
                    r.get('id'), r.get('provider'), r.get('name'), r.get('category'), r.get('country'),
                    r.get('symbol_scope','GLOBAL'), int(r.get('importance') or 0), int(r.get('start_ts')), int(r.get('end_ts')), r.get('source_payload'),
                )
            )
            inserted += 1
    logger.info("[events] upserted {} events", inserted)
    # Invalidate gating window cache so newly inserted events are immediately visible to evaluations
    if inserted:
        try:  # local import to avoid circular at module import time
            from ultra_signals.events import gating  # type: ignore
            gating.reset_caches()
        except Exception:
            pass
    return inserted


def load_events_window(from_ts: int, to_ts: int) -> List[Dict[str, Any]]:
    return fetchall("SELECT * FROM events WHERE end_ts>=? AND start_ts<=? ORDER BY start_ts", (from_ts, to_ts))


__all__ = ["DDL_EVENTS","upsert_events","load_events_window"]
