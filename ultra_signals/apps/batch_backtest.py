"""Batch backtest runner for multi-symbol / multi-timeframe sweeps.

Usage:
  python -m ultra_signals.apps.batch_backtest --config settings.yaml [--force] [--sequential] [--wf-only-top10]

Behavior:
  * Loads batch_run block from settings.yaml
  * Expands jobs (symbol,timeframe) with rule: 5m only for top10, everyone runs 15m & 1h
  * Skips jobs if report.json exists unless --force
  * Runs in parallel (ThreadPool) limited by max_parallel_workers (unless --sequential)
  * Collects KPIs from each report.json -> leaderboard.csv (append/update)
  * Generates summary.md ranked by profit_factor (tie-break sortino desc, max_drawdown_pct asc)
  * Logs errors to errors.log and continues
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

from loguru import logger
from ultra_signals.backtest.json_metrics import CORE_FIELDS, DIM_FIELDS
from ultra_signals.core.config import load_settings

TOP10 = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
    "ADAUSDT","DOGEUSDT","AVAXUSDT","LINKUSDT","MATICUSDT"
]

METRIC_FIELDS = DIM_FIELDS + CORE_FIELDS  # include dimensions + metrics


def ensure_metric_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a metrics dict containing all CORE_FIELDS, filling missing numeric with 0."""
    out = {}
    for f in CORE_FIELDS:
        v = data.get(f)
        if isinstance(v, (int, float)):
            out[f] = v
        else:
            # allow symbol/timeframe pass-through; else default 0
            if f in ("symbol","timeframe") and isinstance(v, str):
                out[f] = v
            else:
                out[f] = 0.0
    # preserve original symbol/timeframe if present
    if 'symbol' in data:
        out['symbol'] = data['symbol']
    if 'timeframe' in data:
        out['timeframe'] = data['timeframe']
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser("Batch Backtest Runner")
    p.add_argument("--config", default="settings.yaml", help="Path to settings file")
    p.add_argument("--force", action="store_true", help="Re-run even if report.json exists")
    p.add_argument("--sequential", action="store_true", help="Run jobs sequentially (debug/memory)")
    p.add_argument("--wf-only-top10", action="store_true", help="When walk_forward.enabled, limit WF jobs to top10 @15m only")
    p.add_argument("--pool", choices=["thread","process"], default="thread", help="Executor pool type for scheduling jobs (subprocess per job regardless).")
    return p.parse_args(argv)


def load_batch_config(path: str) -> Dict[str, Any]:
    settings = load_settings(path)
    # Pydantic model -> dict
    cfg = settings.model_dump() if hasattr(settings, "model_dump") else settings
    batch = cfg.get("batch_run") or {}
    if not batch:
        raise SystemExit("batch_run block missing in settings.yaml")
    return batch


def expand_jobs(batch_cfg: Dict[str, Any], wf_only_top10: bool) -> List[Tuple[str,str,str,str,bool]]:
    symbols: List[str] = batch_cfg.get("symbols") or []
    tfs: List[str] = batch_cfg.get("timeframes") or []
    start = batch_cfg.get("start_date")
    end = batch_cfg.get("end_date")
    wf_enabled = bool((batch_cfg.get("walk_forward") or {}).get("enabled"))

    jobs: List[Tuple[str,str,str,str,bool]] = []  # (symbol, timeframe, start, end, walk_forward)

    for sym in symbols:
        if wf_enabled and wf_only_top10:
            # Only schedule WF for top10 @15m
            if sym in TOP10:
                jobs.append((sym, "15m", start, end, True))
            continue  # skip non-top10
        for tf in tfs:
            if tf == "5m" and sym not in TOP10:
                continue  # enforce 5m only on top10
            if tf == "5m":
                jobs.append((sym, tf, start, end, False if wf_enabled else False))
            elif tf in ("15m","1h"):
                jobs.append((sym, tf, start, end, wf_enabled))
    return jobs


def job_needs_run(out_dir: Path, force: bool) -> bool:
    if force:
        return True
    return not (out_dir / "report.json").exists()


def run_job(config_path: str, job: Tuple[str,str,str,str,bool], base_reports: Path) -> Tuple[str,str,Dict[str,Any],str]:
    symbol, tf, start, end, wf = job
    # Choose output dir root based on wf flag
    root = base_reports / ("batch_wf" if wf else "batch") / f"{symbol}_{tf}"
    root.mkdir(parents=True, exist_ok=True)
    if not job_needs_run(root, force=run_job.force):  # type: ignore[attr-defined]
        try:
            # Load existing JSON
            with open(root / "report.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            return symbol, tf, data, "skipped"
        except Exception:
            pass  # re-run if corrupt
    cmd: List[str]
    if wf:
        cmd = [sys.executable, "-m", "ultra_signals.apps.backtest_cli", "wf", "--config", config_path, "--output-dir", str(root), "--json"]
    else:
        cmd = [
            sys.executable, "-m", "ultra_signals.apps.backtest_cli", "run", "--config", config_path,
            "--symbol", symbol, "--tf", tf, "--from", start, "--to", end, "--output-dir", str(root), "--json"
        ]
    logger.info(f"JOB start {symbol} {tf} wf={wf} -> {root}")
    import subprocess
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"exit={res.returncode} stderr={res.stderr[-400:]}" )
    except Exception as e:
        raise RuntimeError(f"Job {symbol} {tf} failed: {e}")
    # Parse JSON (WF path: might not create report.json; we fabricate minimal metrics)
    report_json = root / "report.json"
    data: Dict[str,Any]
    if report_json.exists():
        try:
            data_raw = json.loads(report_json.read_text(encoding='utf-8'))
        except Exception as e:
            raise RuntimeError(f"Failed reading report.json for {symbol} {tf}: {e}")
    else:
        data_raw = {"symbol":symbol,"timeframe":tf}
    data_norm = ensure_metric_fields(data_raw)
    return symbol, tf, data_norm, "ok"

# attribute to allow dynamic setting in worker
run_job.force = False  # type: ignore


def append_leaderboard_row(leaderboard_path: Path, metrics: Dict[str,Any]):
    import csv
    exists = leaderboard_path.exists()
    with leaderboard_path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if not exists:
            w.writeheader()
        row = {k: metrics.get(k, '') for k in METRIC_FIELDS}
        w.writerow(row)


def build_summary(leaderboard_path: Path, summary_path: Path):
    import pandas as pd
    if not leaderboard_path.exists():
        return
    df = pd.read_csv(leaderboard_path)
    if df.empty:
        return
    # Ranking: PF desc, Sortino desc, MaxDD_pct asc
    df['rank'] = df.sort_values(['profit_factor','sortino','max_drawdown_pct'], ascending=[False,False,True]).reset_index(drop=True).index + 1
    lines = ["# Batch Summary\n", f"Generated: {datetime.utcnow().isoformat()}Z\n\n"]
    # Leaderboard table
    keep_cols = ['rank','symbol','timeframe','profit_factor','sortino','sharpe','max_drawdown_pct','win_rate_pct','total_trades','net_pnl']
    lines.append(df[keep_cols].to_markdown(index=False))
    lines.append("\n\n## Top 5 By Metric\n")
    def top_section(metric: str, asc=False):
        sub = df.sort_values(metric, ascending=asc).head(5)
        lines.append(f"### {metric}\n")
        lines.append(sub[['symbol','timeframe',metric]].to_markdown(index=False))
        lines.append("\n")
    top_section('profit_factor')
    top_section('sortino')
    top_section('max_drawdown_pct', asc=True)
    top_section('win_rate_pct')
    top_section('net_pnl')
    summary_path.write_text("\n".join(lines), encoding='utf-8')


def main(argv=None):
    args = parse_args(argv)
    batch_cfg = load_batch_config(args.config)
    jobs = expand_jobs(batch_cfg, wf_only_top10=args.wf_only_top10)
    max_workers = int(batch_cfg.get('max_parallel_workers', 4))
    base_reports = Path('reports')
    leaderboard_path = base_reports / 'batch' / 'leaderboard.csv'
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    errors_log = base_reports / 'batch' / 'errors.log'
    run_job.force = args.force  # type: ignore

    logger.info(f"Planned jobs: {len(jobs)} (force={args.force} sequential={args.sequential})")

    results: List[Tuple[str,str,Dict[str,Any],str]] = []
    if args.sequential or max_workers <= 1:
        for j in jobs:
            try:
                results.append(run_job(args.config, j, base_reports))
            except Exception as e:
                errors_log.parent.mkdir(parents=True, exist_ok=True)
                with open(errors_log, 'a', encoding='utf-8') as ef:
                    ef.write(f"{j[0]},{j[1]} -> {e}\n")
                logger.error(e)
    else:
        ExecutorCls = ThreadPoolExecutor if args.pool == 'thread' else ProcessPoolExecutor
        with ExecutorCls(max_workers=max_workers) as ex:
            fut_map = {ex.submit(run_job, args.config, j, base_reports): j for j in jobs}
            for fut in as_completed(fut_map):
                j = fut_map[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    errors_log.parent.mkdir(parents=True, exist_ok=True)
                    with open(errors_log, 'a', encoding='utf-8') as ef:
                        ef.write(f"{j[0]},{j[1]} -> {e}\n")
                    logger.error(e)

    # Append leaderboard
    for _, _, metrics, status in results:
        try:
            append_leaderboard_row(leaderboard_path, metrics)
        except Exception as e:
            logger.error(f"Failed to append leaderboard: {e}")

    # Build summary
    try:
        build_summary(leaderboard_path, base_reports / 'batch' / 'summary.md')
        logger.info("summary.md generated")
    except Exception as e:
        logger.error(f"Failed to write summary: {e}")

    logger.success("Batch complete")

if __name__ == "__main__":
    main()
