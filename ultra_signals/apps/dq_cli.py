"""Data Quality CLI (audit & live monitor) – Sprint 39.

Usage examples:
  python -m ultra_signals.apps.dq_cli audit --config settings.yaml --symbols BTCUSDT,ETHUSDT --venue BINANCE --tf 5m --from 2025-01-01 --to 2025-08-01 --out reports/data_quality/audit_2025_01_08
  python -m ultra_signals.apps.dq_cli live --config settings.yaml --symbols BTCUSDT,ETHUSDT --venues BINANCE,BYBIT
"""
from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
from loguru import logger
import pandas as pd
from ..core.config import load_settings
from ..dq import validators, normalizer, time_sync
from ..backtest.data_adapter import DataAdapter


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def cmd_audit(args):  # pragma: no cover (integration oriented)
    settings = load_settings(args.config)
    out_dir = args.out
    _ensure_dir(out_dir)
    symbols = args.symbols.split(',')
    venue = args.venue
    tf = args.tf
    tf_ms = _tf_to_ms(tf)
    adapter = DataAdapter(settings.backtest.data if settings.backtest else {"provider": "csv", "base_path": "data"})  # type: ignore
    summary = []
    coverage_rows = []
    for sym in symbols:
        try:
            df = adapter.load_ohlcv(sym, tf, args.from_, args.to)
            if df is not None and not df.empty:
                work = df.reset_index().rename(columns={df.index.name or 'timestamp': 'timestamp'})
                work['ts'] = pd.to_datetime(work['timestamp']).astype('int64') // 1_000_000
                work = work[['ts','open','high','low','close','volume']]
            else:
                work = pd.DataFrame(columns=['ts','open','high','low','close','volume'])
            rep = validators.validate_ohlcv_df(work, tf_ms, settings.model_dump(), sym, venue)
            summary.append({"symbol": sym, "ok": rep.ok, "errors": rep.errors, "warnings": rep.warnings, **rep.stats})
            if 'coverage_pct' in rep.stats:
                coverage_rows.append({"symbol": sym, "coverage_pct": rep.stats['coverage_pct']})
            if settings.data_quality.get('write_parquet_snaps', True) and not work.empty:  # type: ignore
                try:
                    work.to_parquet(Path(out_dir)/f"{sym}_{tf}.parquet")
                except Exception:
                    pass
        except Exception as e:
            summary.append({"symbol": sym, "ok": False, "errors": [str(e)], "warnings": [], "rows": 0})
    with open(Path(out_dir)/'dq_summary.json','w',encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    if coverage_rows:
        pd.DataFrame(coverage_rows).to_csv(Path(out_dir)/'coverage.csv', index=False)
    logger.info(f"audit.complete out={out_dir} symbols={len(symbols)}")


def cmd_live(args):  # pragma: no cover (integration oriented)
    settings = load_settings(args.config)
    symbols = args.symbols.split(',')
    venues = args.venues.split(',')
    logger.info(f"Starting live DQ monitor symbols={symbols} venues={venues}")
    # Skeleton: user should wire actual polling loop
    try:
        time_sync.assert_within_skew(settings.model_dump(), venues=venues)
        logger.info("initial skew check passed")
    except Exception as e:
        logger.error(f"skew_check_failed err={e}")


def build_parser():
    p = argparse.ArgumentParser("dq_cli")
    sub = p.add_subparsers(dest='cmd', required=True)
    pa = sub.add_parser('audit')
    pa.add_argument('--config', required=True)
    pa.add_argument('--symbols', required=True)
    pa.add_argument('--venue', required=True)
    pa.add_argument('--tf', required=True)
    pa.add_argument('--from', dest='from_', required=True)
    pa.add_argument('--to', dest='to', required=True)
    pa.add_argument('--out', required=True)
    pa.set_defaults(func=cmd_audit)
    pl = sub.add_parser('live')
    pl.add_argument('--config', required=True)
    pl.add_argument('--symbols', required=True)
    pl.add_argument('--venues', required=True)
    pl.set_defaults(func=cmd_live)
    return p

def _tf_to_ms(tf: str) -> int:
    if tf.endswith('ms'): return int(tf[:-2])
    if tf.endswith('s'): return int(tf[:-1]) * 1000
    if tf.endswith('m'): return int(tf[:-1]) * 60_000
    if tf.endswith('h'): return int(tf[:-1]) * 3_600_000
    if tf.endswith('d'): return int(tf[:-1]) * 86_400_000
    return 0


def main(argv=None):  # pragma: no cover
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == '__main__':  # pragma: no cover
    main()
