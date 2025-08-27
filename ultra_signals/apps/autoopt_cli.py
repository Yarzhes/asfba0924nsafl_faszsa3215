"""CLI entrypoint for Auto Optimization (Sprint 27).

Usage:
  python -m ultra_signals.apps.autoopt_cli run --symbols BTCUSDT --tfs 5m --profiles trend --trials 10 --output runs/demo
  python -m ultra_signals.apps.autoopt_cli promote --run runs/demo --yes
  python -m ultra_signals.apps.autoopt_cli compare --a baseline.yaml --b runs/demo/champion.yaml
"""
from __future__ import annotations
import argparse, json, sys, time, copy, csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from loguru import logger
from ultra_signals.autoopt.spaces import AutoOptSpace
from ultra_signals.autoopt.objective import compute_risk_aware_score
from ultra_signals.autoopt.wf_runner import AutoOptWFRunner
from ultra_signals.autoopt.selection import rank_candidates
from ultra_signals.autoopt.publisher import publish_champion

# NOTE: We reuse existing data adapter / engine factories where possible to avoid duplication.
try:
    from ultra_signals.backtest.data_adapter import DataAdapter
    from ultra_signals.engine.real_engine import RealSignalEngine
except Exception:  # pragma: no cover
    DataAdapter = None
    RealSignalEngine = None


def _load_settings_from_file(path: str|Path) -> dict:
    import yaml
    with open(path,'r') as f:
        return yaml.safe_load(f)


def _apply_params_to_settings(base_settings: dict, params: dict) -> dict:
    derived = copy.deepcopy(base_settings)
    ens = derived.setdefault('ensemble', {})
    ens['vote_threshold'] = {
        'trend': params['ensemble.vote_threshold_trend'],
        'mean_revert': params['ensemble.vote_threshold_mean_revert']
    }
    ens['min_agree'] = {
        'trend': params['ensemble.min_agree_trend'],
        'mean_revert': params['ensemble.min_agree_mean_revert']
    }
    ens['margin_of_victory'] = params['ensemble.margin_of_victory']
    ens['confidence_floor'] = params['ensemble.confidence_floor']
    return derived


def _trial_worker(serialized: str) -> dict:
    payload = json.loads(serialized)
    settings = payload['settings']
    params = payload['params']
    symbol = payload['symbol']; tf = payload['tf']; profile = payload['profile']
    derived = _apply_params_to_settings(settings, params)
    runner = AutoOptWFRunner(settings, lambda s: DataAdapter(s), lambda s: RealSignalEngine(s, None))
    metrics = runner.evaluate(derived, symbol, tf)
    score = compute_risk_aware_score(metrics)
    return {'score': score, **metrics, **params, 'symbol': symbol, 'timeframe': tf, 'profile': profile}


def _load_baseline_score(baseline_path: Path) -> float:
    if not baseline_path.exists():
        return 0.0
    try:
        data = json.loads(baseline_path.read_text())
        return float(data.get('score', 0.0))
    except Exception:
        return 0.0


def cmd_run(args):
    settings = _load_settings_from_file(args.settings)
    space = AutoOptSpace()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    symbols = args.symbols.split(',')
    tfs = args.tfs.split(',')
    profiles = args.profiles.split(',')
    grid = [(s, tf, pr) for s in symbols for tf in tfs for pr in profiles]
    logger.info(f"Grid size {len(grid)} (symbols×tfs×profiles)")
    baseline_score = _load_baseline_score(Path(args.baseline)) if args.baseline else 0.0
    logger.info(f"Baseline score={baseline_score}")
    leaderboard_rows = []
    # sample parameter sets once then evaluate across grid for each trial
    class DummyTrial: pass
    futures = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        for trial in range(args.trials):
            params = space.sample(DummyTrial())
            for (symbol, tf, profile) in grid:
                payload = json.dumps({'settings': settings, 'params': params, 'symbol': symbol, 'tf': tf, 'profile': profile})
                fut = ex.submit(_trial_worker, payload)
                fut.trial_index = trial  # type: ignore
                futures.append(fut)
        for fut in as_completed(futures):
            res = fut.result()
            res['trial'] = getattr(fut,'trial_index', -1)
            leaderboard_rows.append(res)
    dur = time.time() - start
    if not leaderboard_rows:
        logger.error('No results produced')
        return 2
    # selection uses best per symbol/tf/profile (highest score for that tuple)
    best_by_key = {}
    for r in leaderboard_rows:
        key = (r['symbol'], r['timeframe'], r['profile'])
        if key not in best_by_key or r['score'] > best_by_key[key]['score']:
            best_by_key[key] = r
    ranked_input = list(best_by_key.values())
    sel = rank_candidates(ranked_input, args.min_uplift, baseline_score)
    fieldnames = sorted(leaderboard_rows[0].keys())
    with open(output/'leaderboard.csv','w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(leaderboard_rows)
    # write challengers individually for audit
    challengers_dir = output/'challengers'; challengers_dir.mkdir(exist_ok=True)
    import yaml
    for i,row in enumerate(ranked_input):
        (challengers_dir/f'challenger_{i+1}.yaml').write_text(yaml.safe_dump(row))
    if sel['champion']:
        (output/'champion.json').write_text(json.dumps(sel['champion'], indent=2))
    meta = {'grid': len(grid), 'trials': args.trials, 'baseline_score': baseline_score, 'duration_sec': round(dur,2)}
    (output/'meta.json').write_text(json.dumps(meta, indent=2))
    logger.info(f"Run complete. Trials={args.trials} grid={len(grid)} rows={len(leaderboard_rows)} champion={bool(sel['champion'])} dur={dur:.1f}s")


def cmd_promote(args):
    run_dir = Path(args.run)
    champ_file = run_dir / 'champion.json'
    if not champ_file.exists():
        logger.error("Champion not found; run autoopt first.")
        return 2
    champ = json.loads(champ_file.read_text())
    publish_champion(Path('.'), champ.get('symbol','BTCUSDT'), champ.get('timeframe','5m'), champ.get('profile','trend'), champ, champ, champ)
    logger.info("Champion promoted.")


def cmd_compare(args):
    import yaml
    a = _load_settings_from_file(args.a)
    b = _load_settings_from_file(args.b)
    diff_keys = []
    for k in set(a.keys()).union(b.keys()):
        if a.get(k) != b.get(k):
            diff_keys.append(k)
    print("Changed top-level keys:", diff_keys)


def build_parser():
    p = argparse.ArgumentParser('autoopt')
    sub = p.add_subparsers(dest='cmd')
    r = sub.add_parser('run')
    r.add_argument('--symbols', required=True)
    r.add_argument('--tfs', required=True)
    r.add_argument('--profiles', required=True)
    r.add_argument('--trials', type=int, default=20)
    r.add_argument('--output', required=True)
    r.add_argument('--settings', default='settings.yaml')
    r.add_argument('--min-uplift', type=float, default=0.07, dest='min_uplift')
    r.add_argument('--baseline', help='Path to baseline champion.json for uplift comparison')
    r.add_argument('--max-workers', type=int, default=4)
    r.set_defaults(func=cmd_run)
    prm = sub.add_parser('promote')
    prm.add_argument('--run', required=True)
    prm.add_argument('--yes', action='store_true')
    prm.set_defaults(func=cmd_promote)
    cmp = sub.add_parser('compare')
    cmp.add_argument('--a', required=True)
    cmp.add_argument('--b', required=True)
    cmp.set_defaults(func=cmd_compare)
    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    if not hasattr(args,'func'):
        p.print_help(); return 1
    return args.func(args)

if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
