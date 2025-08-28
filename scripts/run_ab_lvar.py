import yaml
import os
import sys
from types import SimpleNamespace
from ultra_signals.core.config import load_settings
from ultra_signals.apps.backtest_cli import handle_wf

ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG = os.path.join(ROOT, 'backtests', 'ab_lvar_experiment.yaml')


def _make_args(output_dir: str):
    # Minimal args object expected by handle_wf: output_dir, profiles, hot_reload, symbols, json
    return SimpleNamespace(output_dir=output_dir, profiles=None, hot_reload=False, symbols=None, json=True)


def main():
    with open(CONFIG, 'r') as f:
        cfg = yaml.safe_load(f)
    variants = cfg.get('variants', [])
    common = cfg.get('common', {}) or {}
    print(f"Found experiment: {cfg.get('experiment_name')} with {len(variants)} variants")

    results = {}
    base_settings = load_settings(common.get('base_config', 'settings.yaml')) if common else load_settings()

    for v in variants:
        name = v.get('name')
        risk = v.get('risk', {})
        print(f"Running variant: {name} -> overrides risk={risk}")
        # Apply overrides to a fresh copy of base settings dict
        settings_dict = base_settings.model_dump() if hasattr(base_settings, 'model_dump') else dict(base_settings)
        settings_dict.setdefault('risk', {})
        settings_dict['risk'].update(risk or {})

        # Prepare output dir per variant
        out_dir = os.path.join('reports', f"ab_{cfg.get('experiment_name')}_{name}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        args = _make_args(output_dir=out_dir)

        # Call the walk-forward handler which will write report.json into out_dir
        try:
            handle_wf(args, settings_dict)
        except Exception as e:
            print(f"Variant {name} failed: {e}")
            results[name] = {'error': str(e)}
            continue

        # Collect produced report.json if present
        report_path = os.path.join(out_dir, 'report.json')
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as rf:
                    import json
                    payload = json.load(rf)
                    results[name] = payload
            except Exception as e:
                results[name] = {'error': f'failed to read report.json: {e}'}
        else:
            results[name] = {'error': 'no report.json produced'}

    # Print compact summary
    print('\nA/B Results:')
    for k, v in results.items():
        if 'error' in v:
            print(f"- {k}: ERROR -> {v.get('error')}")
        else:
            sharpe = v.get('sharpe') or v.get('sharpe_ratio') or v.get('sharpe', 0.0)
            maxdd = v.get('max_drawdown_pct') or v.get('max_drawdown') or v.get('max_drawdown_pct', 0.0)
            print(f"- {k}: sharpe={sharpe} maxdd={maxdd}")


if __name__ == '__main__':
    main()
