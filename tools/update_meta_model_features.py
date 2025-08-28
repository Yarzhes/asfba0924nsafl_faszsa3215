"""Update a meta-scorer joblib bundle's feature_names by adding new sentiment/topic fields.

Usage:
  Set PYTHONPATH to repo root if needed and run:
    $env:PYTHONPATH='C:\\Users\\Almir\\Projects\\Trading Helper'; python tools/update_meta_model_features.py [model_path]

Default model_path: calibration_model.joblib
"""
import sys
from pathlib import Path
import json

DEFAULT = 'calibration_model.joblib'
NEW_FEATURES = [
    'sent_topic_etf_regulation_score_s',
    'sent_topic_etf_regulation_z',
    'sent_topic_etf_regulation_pctl',
    'sent_topic_hack_exploit_score_s',
    'sent_topic_meme_retail_score_s',
    'sent_vs_funding_div_long',
    'sent_vs_funding_div_short',
    'contrarian_flag_long',
    'contrarian_flag_short'
]


def main(model_path: str):
    try:
        import joblib
    except Exception as e:
        print('joblib not available:', e)
        return 2
    p = Path(model_path)
    if not p.exists():
        print(f'model file not found: {model_path}')
        return 1
    try:
        bundle = joblib.load(str(p))
    except Exception as e:
        print('failed loading model:', e)
        return 3
    if not isinstance(bundle, dict):
        print('model bundle is not a dict; aborting to avoid corruption')
        return 4
    feats = list(bundle.get('feature_names') or [])
    before = len(feats)
    added = []
    for f in NEW_FEATURES:
        if f not in feats:
            feats.append(f)
            added.append(f)
    bundle['feature_names'] = feats
    try:
        joblib.dump(bundle, str(p))
    except Exception as e:
        print('failed saving model bundle:', e)
        return 5
    print(f'Updated model: {model_path}\nBefore: {before} features\nAdded: {len(added)} features\nNew total: {len(feats)}')
    if added:
        print('Added features:')
        print(json.dumps(added, indent=2))
    return 0


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    sys.exit(main(path))
