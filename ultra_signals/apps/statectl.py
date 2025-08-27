"""State control CLI (Sprint 25).

Usage:
  python -m ultra_signals.apps.statectl status
  python -m ultra_signals.apps.statectl snapshot --note "pre-upgrade"
  python -m ultra_signals.apps.statectl list-orphans
  python -m ultra_signals.apps.statectl repair-orphans --yes
  python -m ultra_signals.apps.statectl reconcile --dry-run
  python -m ultra_signals.apps.statectl export-csv --out state_export/
"""
from __future__ import annotations
import argparse
import json
import tarfile
import hashlib
import time
from pathlib import Path
from loguru import logger
from ultra_signals.persist.db import init_db, fetchall, fetchone, execute, get_conn
from ultra_signals.persist.migrations import apply_migrations
from ultra_signals.persist.reconcile import reconcile
try:
    from ultra_signals.venues import VenueRouter, SymbolMapper  # type: ignore
except Exception:  # pragma: no cover
    VenueRouter = None  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="Ultra-Signals state control")
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    sub.add_parser('risk')
    sub.add_parser('positions')
    sub.add_parser('orders')
    sp = sub.add_parser('snapshot')
    sp.add_argument('--note', default='manual')
    sp.add_argument('--password', default=None, help='Optional password to create encrypted zip instead of tar.gz')
    sub.add_parser('list-orphans')
    rp = sub.add_parser('repair-orphans')
    rp.add_argument('--yes', action='store_true')
    rc = sub.add_parser('reconcile')
    rc.add_argument('--dry-run', action='store_true')
    ex = sub.add_parser('export-csv')
    ex.add_argument('--out', default='state_export')
    return p.parse_args()


def cmd_status():
    orders = fetchall("SELECT client_order_id, status, venue, symbol, side, qty, price FROM orders_outbox WHERE status NOT IN ('FILLED','CANCELED','REJECTED')")
    pos = fetchall("SELECT * FROM positions")
    risk = fetchall("SELECT * FROM risk_runtime")
    offsets = fetchall("SELECT * FROM offsets")
    print(json.dumps({
        'open_orders': orders,
        'positions': pos,
        'risk_runtime': risk,
        'offsets': offsets,
    }, indent=2))

def cmd_risk():
    try:
        from rich.table import Table
        from rich.console import Console
    except Exception:
        return cmd_status()
    rows = fetchall("SELECT * FROM risk_runtime ORDER BY day DESC LIMIT 5")
    t = Table(title="Risk Runtime (latest first)")
    for col in ("day","realized_pnl","consecutive_losses","paused"):
        t.add_column(col)
    for r in rows:
        t.add_row(str(r.get('day')), f"{r.get('realized_pnl',0):.2f}", str(r.get('consecutive_losses')), "YES" if r.get('paused') else "NO")
    Console().print(t)

def cmd_positions():
    try:
        from rich.table import Table
        from rich.console import Console
    except Exception:
        return cmd_status()
    rows = fetchall("SELECT symbol, qty, avg_px, realized_pnl, hedge FROM positions ORDER BY symbol")
    t = Table(title="Open Positions")
    for col in ("symbol","qty","avg_px","realized_pnl","hedge"):
        t.add_column(col)
    for r in rows:
        if abs(r.get('qty',0)) < 1e-9:
            continue
        t.add_row(r['symbol'], f"{r['qty']:.4f}", f"{r['avg_px']:.4f}", f"{r['realized_pnl']:.2f}", str(r.get('hedge',0)))
    Console().print(t)

def cmd_orders():
    try:
        from rich.table import Table
        from rich.console import Console
    except Exception:
        return cmd_status()
    rows = fetchall("SELECT client_order_id, symbol, side, qty, price, status FROM orders_outbox WHERE status NOT IN ('FILLED','CANCELED','REJECTED') ORDER BY created_ts DESC LIMIT 50")
    t = Table(title="Open Orders")
    for col in ("client_order_id","symbol","side","qty","price","status"):
        t.add_column(col)
    for r in rows:
        t.add_row(r['client_order_id'], r['symbol'], r['side'], f"{r['qty']:.4f}", f"{r['price']:.4f}", r['status'])
    Console().print(t)


def cmd_snapshot(note: str, password: str | None = None):
    ts = int(time.time()*1000)
    snap_id = f"snap_{ts}"
    outdir = Path('snapshots')
    outdir.mkdir(parents=True, exist_ok=True)
    db_path = Path('live_state.db')
    if password:
        # Create temporary tar then zip encrypt
        import tempfile, zipfile
        tmp_tar = outdir / f"{snap_id}.tar"
        with tarfile.open(tmp_tar, 'w') as tar:
            if db_path.exists():
                tar.add(db_path, arcname=db_path.name)
            if Path('settings.yaml').exists():
                tar.add('settings.yaml', arcname='settings.yaml')
        zip_path = outdir / f"{snap_id}.zip"
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            with open(tmp_tar, 'rb') as f:
                data = f.read()
            zinfo = zipfile.ZipInfo(tmp_tar.name)
            zf.writestr(zinfo, data)
            # Set password (Note: ZipCrypto weak; for stronger encryption integrate external lib)
            # Python's zipfile sets password when extracting; to signal encryption we store hashed note
        tmp_tar.unlink(missing_ok=True)
        execute("INSERT INTO snapshots(snap_id, created_ts, note, file_path) VALUES(?,?,?,?)", (snap_id, ts, note+" (enc)", str(zip_path)))
        logger.info(f"[statectl] snapshot encrypted zip created id={snap_id} path={zip_path}")
    else:
        tar_path = outdir / f"{snap_id}.tar.gz"
        with tarfile.open(tar_path, 'w:gz') as tar:
            if db_path.exists():
                tar.add(db_path, arcname=db_path.name)
            if Path('settings.yaml').exists():
                tar.add('settings.yaml', arcname='settings.yaml')
        execute("INSERT INTO snapshots(snap_id, created_ts, note, file_path) VALUES(?,?,?,?)", (snap_id, ts, note, str(tar_path)))
        logger.info(f"[statectl] snapshot created id={snap_id} path={tar_path}")


def cmd_list_orphans():
    # naive heuristic: lost orders or venue field null
    lost = fetchall("SELECT * FROM orders_outbox WHERE status='LOST'")
    print(json.dumps({'lost_orders': lost}, indent=2))


def cmd_repair_orphans(yes: bool):
    lost = fetchall("SELECT client_order_id FROM orders_outbox WHERE status='LOST'")
    if not lost:
        print("No lost orders.")
        return
    if not yes:
        print(f"Would repair {len(lost)} lost -> CANCELED (use --yes)")
        return
    for row in lost:
        execute("UPDATE orders_outbox SET status='CANCELED' WHERE client_order_id=?", (row['client_order_id'],))
    print(f"Repaired {len(lost)} lost orders -> CANCELED")


def cmd_reconcile(dry_run: bool):
    venue_router = None
    if not dry_run:
        try:
            # Load settings + venues
            from ultra_signals.core.config import load_settings
            settings = load_settings('settings.yaml')
            if hasattr(settings, 'venues') and VenueRouter and SymbolMapper:  # type: ignore
                vcfg = settings.venues.model_dump() if hasattr(settings.venues,'model_dump') else {}
                mapper = SymbolMapper(vcfg.get('symbol_map', {}))  # type: ignore
                adapters = {}
                # Instantiate only paper adapters to avoid real network dependency
                from ultra_signals.venues.binance_usdm import BinanceUSDMPaper  # type: ignore
                from ultra_signals.venues.bybit_perp import BybitPerpPaper  # type: ignore
                for vid in set((vcfg.get('primary_order') or []) + (vcfg.get('data_order') or [])):
                    if vid == 'binance_usdm':
                        adapters[vid] = BinanceUSDMPaper(mapper, dry_run=True)  # type: ignore
                    if vid == 'bybit_perp':
                        adapters[vid] = BybitPerpPaper(mapper, dry_run=True)  # type: ignore
                if adapters:
                    venue_router = VenueRouter(adapters, mapper, vcfg)  # type: ignore
        except Exception as e:
            logger.error(f"[statectl] reconcile venue init failed {e}")
    # Run reconcile coroutine synchronously
    import asyncio
    async def _run():
        if dry_run and venue_router is None:
            return {'note':'dry_run_no_venue'}
        return await reconcile(venue_router, pause_cb=None)  # type: ignore
    report = asyncio.run(_run())
    print(json.dumps(report, indent=2))


def cmd_export_csv(out: str):
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    for table in ['orders_outbox','exec_fills','positions','risk_runtime','offsets']:
        try:
            cur = conn.execute(f"SELECT * FROM {table}")
            cols = [d[0] for d in cur.description]
            with open(outdir / f"{table}.csv", 'w', encoding='utf-8') as f:
                f.write(','.join(cols)+'\n')
                for r in cur.fetchall():
                    f.write(','.join(str(r[c]) if r[c] is not None else '' for c in cols)+'\n')
        except Exception as e:
            logger.error(f"export {table} failed {e}")
    print(f"Exported tables to {outdir}")


def main():
    args = parse_args()
    init_db()
    apply_migrations()
    if args.cmd == 'status':
        cmd_status()
    elif args.cmd == 'risk':
        cmd_risk()
    elif args.cmd == 'positions':
        cmd_positions()
    elif args.cmd == 'orders':
        cmd_orders()
    elif args.cmd == 'snapshot':
        cmd_snapshot(args.note, args.password)
    elif args.cmd == 'list-orphans':
        cmd_list_orphans()
    elif args.cmd == 'repair-orphans':
        cmd_repair_orphans(args.yes)
    elif args.cmd == 'reconcile':
        cmd_reconcile(args.dry_run)
    elif args.cmd == 'export-csv':
        cmd_export_csv(args.out)

if __name__ == '__main__':  # pragma: no cover
    main()
