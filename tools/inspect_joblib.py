"""Inspect a joblib file and print its top-level type/keys/attributes.

Usage:
  python tools/inspect_joblib.py [path]
"""
import sys
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else 'calibration_model.joblib'
try:
    import joblib
except Exception as e:
    print('joblib missing:', e)
    sys.exit(2)

p = Path(path)
if not p.exists():
    print('missing', path)
    sys.exit(1)

try:
    obj = joblib.load(str(p))
except Exception as e:
    print('load_error', e)
    sys.exit(3)

print('type:', type(obj))
if isinstance(obj, dict):
    print('keys:', list(obj.keys()))
else:
    # try to probe common attributes
    attrs = []
    for a in ['feature_names','clf','model','pre','calibrator']:
        try:
            v = getattr(obj, a, None)
            if v is not None:
                attrs.append((a, type(v)))
        except Exception:
            pass
    if attrs:
        print('attrs:', attrs)
    else:
        try:
            print('repr:', repr(obj)[:800])
        except Exception:
            print('no inspectable attrs')

sys.exit(0)
