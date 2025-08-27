"""Database migrations for Sprint 25 persistence layer."""
from __future__ import annotations
from typing import List
from loguru import logger
from .db import get_conn, tx, fetchall

MIGRATIONS: List[tuple[str,str]] = [
    (
        '0001_base_schema',
        """
        CREATE TABLE IF NOT EXISTS migrations(version TEXT PRIMARY KEY, applied_ts INTEGER);
        CREATE TABLE IF NOT EXISTS instances(instance_id TEXT PRIMARY KEY, started_ts INT, hostname TEXT, pid INT, active INT);
        CREATE TABLE IF NOT EXISTS orders_outbox(
            client_order_id TEXT PRIMARY KEY,
            venue TEXT, symbol TEXT, side TEXT, type TEXT, qty REAL, price REAL,
            reduce_only INT, parent_id TEXT, status TEXT, venue_order_id TEXT,
            last_error TEXT, retries INT, created_ts INT, updated_ts INT,
            profile_id TEXT, cfg_hash TEXT
        );
        CREATE TABLE IF NOT EXISTS exec_fills(
            fill_id TEXT PRIMARY KEY, client_order_id TEXT, venue TEXT, venue_order_id TEXT,
            symbol TEXT, qty REAL, price REAL, fee REAL, is_maker INT, ts INT
        );
        CREATE TABLE IF NOT EXISTS positions(
            symbol TEXT PRIMARY KEY, qty REAL, avg_px REAL, realized_pnl REAL,
            updated_ts INT, venue TEXT NULL, hedge INT DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS risk_runtime(
            day TEXT PRIMARY KEY, realized_pnl REAL, consecutive_losses INT, paused INT
        );
        CREATE TABLE IF NOT EXISTS offsets(
            stream TEXT PRIMARY KEY, last_ts INT, last_seq INT
        );
        CREATE TABLE IF NOT EXISTS snapshots(
            snap_id TEXT PRIMARY KEY, created_ts INT, note TEXT, file_path TEXT
        );
        CREATE TABLE IF NOT EXISTS settings_fingerprint(
            id INT PRIMARY KEY CHECK(id=1), cfg_hash TEXT, profile_version TEXT
        );
        CREATE TABLE IF NOT EXISTS oco_links(
            parent_client_id TEXT PRIMARY KEY, stop_client_id TEXT, tp_client_ids TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_fills_client ON exec_fills(client_order_id);
        CREATE INDEX IF NOT EXISTS ix_orders_status ON orders_outbox(status);
        CREATE INDEX IF NOT EXISTS ix_positions_venue ON positions(venue);
        """
    ),
    (
        '0002_alerts_equity_curve',
        """
        CREATE TABLE IF NOT EXISTS alerts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INT NOT NULL,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT DEFAULT 'INFO',
            meta_json TEXT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_alerts_ts ON alerts(ts);
        CREATE TABLE IF NOT EXISTS equity_curve(
            ts INT PRIMARY KEY,
            equity REAL NOT NULL,
            drawdown REAL NOT NULL
        );
        """
    ),
        (
                '0003_events_table',
                """
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
        ),
    (
        '0004_liquidity_decisions',
        """
        CREATE TABLE IF NOT EXISTS liquidity_decisions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INT NOT NULL,
            symbol TEXT NOT NULL,
            profile TEXT NULL,
            action TEXT NOT NULL,
            reason TEXT NULL,
            spread_bps REAL NULL,
            impact_50k REAL NULL,
            dr REAL NULL,
            rv_5s REAL NULL,
            source TEXT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_liq_decisions_ts ON liquidity_decisions(ts);
        CREATE INDEX IF NOT EXISTS ix_liq_decisions_symbol ON liquidity_decisions(symbol);
        """
    ),
]


def applied_versions() -> set[str]:
    try:
        rows = fetchall("SELECT version FROM migrations")
        return {r['version'] for r in rows}
    except Exception:
        return set()


def apply_migrations():
    conn = get_conn()
    # Ensure migrations table exists
    with tx() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS migrations(version TEXT PRIMARY KEY, applied_ts INTEGER)")
    done = applied_versions()
    for version, ddl in MIGRATIONS:
        if version in done:
            continue
        logger.info(f"[DB] Applying migration {version}")
        with tx() as cur:
            for stmt in filter(None, map(str.strip, ddl.split(';'))):
                if not stmt:
                    continue
                try:
                    cur.execute(stmt)
                except Exception as e:
                    # Backwards compat: older DB may have positions table without 'venue' column; index creation will fail.
                    if 'ix_positions_venue' in stmt and 'no such column: venue' in str(e).lower():
                        try:
                            logger.warning("[DB] Adding missing 'venue' column to positions for backward compatibility")
                            cur.execute("ALTER TABLE positions ADD COLUMN venue TEXT NULL")
                            cur.execute(stmt)  # retry index creation
                            continue
                        except Exception as e2:  # pragma: no cover
                            logger.error(f"[DB] failed to add venue column retroactively: {e2}")
                            raise
                    raise
            cur.execute("INSERT INTO migrations(version, applied_ts) VALUES(?, strftime('%s','now')*1000)", (version,))
    # Compatibility column additions for legacy StateStore expectations
    try:
        cur = conn.execute("PRAGMA table_info(orders_outbox)")
        cols = {r[1] for r in cur.fetchall()}
        compat_cols = {
            'ts': "ALTER TABLE orders_outbox ADD COLUMN ts INT",
            'exchange_order_id': "ALTER TABLE orders_outbox ADD COLUMN exchange_order_id TEXT",
            'filled_qty': "ALTER TABLE orders_outbox ADD COLUMN filled_qty REAL",
            'exec_price': "ALTER TABLE orders_outbox ADD COLUMN exec_price REAL",
            'venue_id': "ALTER TABLE orders_outbox ADD COLUMN venue_id TEXT",
        }
        for col, stmt in compat_cols.items():
            if col not in cols:
                try:
                    with tx() as curx:
                        curx.execute(stmt)
                    logger.info(f"[DB] Added compat column {col} to orders_outbox")
                except Exception as e:  # pragma: no cover
                    logger.warning(f"[DB] compat column add failed {col} {e}")
    except Exception:
        pass
    logger.info("[DB] Migrations complete")

__all__ = ["apply_migrations"]
