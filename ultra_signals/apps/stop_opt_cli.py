"""CLI for Stop Optimization (Sprint 37)

Example:
python -m ultra_signals.apps.stop_opt_cli run --config settings.yaml --from 2024-01-01 --to 2024-08-01 --symbols BTCUSDT,ETHUSDT --out models/stop_opt/stop_table.yaml
"""
from __future__ import annotations
import argparse, yaml, json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from loguru import logger
from ultra_signals.opt.stop_optimizer import optimize, write_table, plot_reports, compute_before_after, write_dated_table, diff_tables


def _load_settings(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) if Path(path).exists() else {}


def _load_trades(symbols: List[str], start: str, end: str, settings: Dict[str, Any]) -> pd.DataFrame:
    # Placeholder: attempt to load trades CSVs under reports/<symbol>_*_trades.csv
    rows = []
    for sym in symbols:
        for p in Path('reports').glob(f'*{sym}*trades.csv'):
            try:
                df = pd.read_csv(p)
                df['symbol'] = sym
                if 'tf' not in df.columns:
                    # infer timeframe from filename heuristically
                    for tf in settings.get('runtime', {}).get('timeframes', ['5m','15m']):
                        if f'_{tf}_' in p.name or p.name.endswith(f'_{tf}.csv'):
                            df['tf'] = tf
                    if 'tf' not in df.columns:
                        df['tf'] = settings.get('runtime', {}).get('primary_timeframe','5m')
                # regime placeholder if missing
                if 'regime' not in df.columns:
                    df['regime'] = 'mixed'
                rows.append(df)
            except Exception as e:
                logger.warning(f'Failed loading {p}: {e}')
    if not rows:
        return pd.DataFrame(columns=['symbol','tf','regime'])
    all_df = pd.concat(rows, ignore_index=True)
    # filter by date if ts_entry present
    if 'ts_entry' in all_df.columns:
        try:
            all_df = all_df.sort_values('ts_entry')
        except Exception:
            pass
    return all_df


def _build_ohlc_lookup(feature_store_pickle: str = None):
    # Placeholder: expects bars cached as parquet under data/ohlcv/<symbol>_<tf>.parquet with datetime index.
    from pathlib import Path
    import pandas as pd
    fs_obj = None
    if feature_store_pickle:
        try:
            import pickle
            with open(feature_store_pickle,'rb') as f:
                fs_obj = pickle.load(f)
        except Exception as e:  # pragma: no cover
            logger.warning(f'Failed loading feature_store pickle: {e}')
    cache: Dict[str,pd.DataFrame] = {}
    def loader(symbol: str, tf: str, ts_start: int, ts_end: int):
        # Prefer live FeatureStore slice if available
        if fs_obj is not None and hasattr(fs_obj,'get_ohlcv_slice'):
            try:
                bars = fs_obj.get_ohlcv_slice(symbol, tf, ts_start, ts_end)
                if bars is not None:
                    return bars
            except Exception:
                pass
        key = f"{symbol}_{tf}"
        if key not in cache:
            p = Path('data/ohlcv')/f'{key}.parquet'
            if p.exists():
                try:
                    cache[key] = pd.read_parquet(p)
                except Exception:
                    cache[key] = pd.DataFrame()
            else:
                cache[key] = pd.DataFrame()
        df = cache[key]
        if df.empty:
            return df
        # assume index is datetime; convert ts (epoch seconds) to datetime
        import pandas as pd
        start_dt = pd.to_datetime(ts_start, unit='s')
        end_dt = pd.to_datetime(ts_end, unit='s')
        return df[(df.index>=start_dt) & (df.index<=end_dt)]
    return loader


def run(args):
    settings = _load_settings(args.config)
    symbols = [s.strip() for s in args.symbols.split(',')] if args.symbols else settings.get('runtime',{}).get('symbols', [])
    trades = _load_trades(symbols, args.start, args.end, settings)
    if trades.empty:
        logger.warning('No trades loaded; aborting optimization.')
        return
    # override objective if flag provided
    if args.objective:
        settings.setdefault('auto_stop_opt', {})['objective'] = args.objective
    ohlc_lookup = _build_ohlc_lookup(args.feature_store_pickle) if args.use_ohlc else None
    table = optimize(trades, settings, return_candidates=True, save_candidates_dir=(args.save_candidates or None), ohlc_lookup=ohlc_lookup)
    out_path = args.out or (settings.get('auto_stop_opt') or {}).get('output_path') or 'models/stop_opt/stop_table.yaml'
    dated_path = write_dated_table(table['table'], out_path)
    logger.info(f'Wrote stop table to {out_path} and dated copy {dated_path}')
    if args.plot and table.get('candidates'):
        plot_reports(table['candidates'], Path(out_path).parent.as_posix())
    # before/after comparison vs baseline ATR multiplier (from risk.adaptive_exits)
    base_mult = float(((settings.get('risk') or {}).get('adaptive_exits') or {}).get('atr_mult_stop', 1.2))
    comparison = compute_before_after(trades, table['table'], base_mult, bootstrap=args.bootstrap, iters=args.bootstrap_iters)
    if not comparison.empty:
        comp_path = Path(out_path).parent / 'before_after_comparison.csv'
        comparison.to_csv(comp_path, index=False)
        logger.info(f'Wrote before/after comparison to {comp_path}')
        if args.plot:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6,4))
                comparison.sort_values('delta_expectancy', ascending=False, inplace=True)
                plt.bar(range(len(comparison)), comparison['delta_expectancy'])
                plt.xticks(range(len(comparison)), [f"{r.symbol}\n{r.tf}\n{r.regime}" for r in comparison.itertuples()], rotation=45, ha='right')
                plt.ylabel('Delta Expectancy')
                plt.title('Before/After Expectancy Improvement')
                plt.tight_layout(); plt.savefig(Path(out_path).parent / 'before_after_expectancy.png'); plt.close()
            except Exception:
                pass
    # shadow diff if previous table exists (last dated)
    historical = sorted(Path(out_path).parent.glob('stop_table_*.yaml'))
    if len(historical) >= 2:
        import yaml as _y
        old = _y.safe_load(historical[-2].read_text()) or {}
        changes = diff_tables(old, table['table'])
        changes_path = Path(out_path).parent / 'stop_table_changes.json'
        import json
        changes_path.write_text(json.dumps(changes, indent=2))
        logger.info(f'Change summary written to {changes_path}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')
    r = sub.add_parser('run')
    r.add_argument('--config', required=True)
    r.add_argument('--from', dest='start', required=False)
    r.add_argument('--to', dest='end', required=False)
    r.add_argument('--symbols', required=False)
    r.add_argument('--out', required=False)
    r.add_argument('--objective', required=False, choices=['expectancy','calmar','sortino'])
    r.add_argument('--plot', action='store_true')
    r.add_argument('--use-ohlc', action='store_true', help='Enable OHLC micro replay using data/ohlcv cache')
    r.add_argument('--bootstrap', action='store_true', help='Compute bootstrap p-values for expectancy improvement')
    r.add_argument('--bootstrap-iters', type=int, default=300)
    r.add_argument('--save-candidates', dest='save_candidates', required=False)
    r.add_argument('--feature-store-pickle', required=False, help='Path to pickled FeatureStore for direct OHLC slices')
    args = ap.parse_args()
    if args.cmd == 'run':
        run(args)
    else:
        ap.print_help()

if __name__ == '__main__':
    main()
