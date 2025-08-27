import json, numpy as np
from pathlib import Path

def test_feature_contract_no_leakage(tmp_path):
    # Assume dataset already built in default path; if not, skip gracefully
    data_dir = Path('data/meta_dataset')  # conventional path (adjust if needed)
    features = data_dir / 'features.npy'
    names = data_dir / 'feature_names.json'
    if not features.exists() or not names.exists():
        return  # skip (dataset not built in test env)
    fn = json.loads(names.read_text())
    leak_terms = [n for n in fn if any(x in n.lower() for x in ['meta_gate','prob','calibrated_prob','p_'])]
    assert not leak_terms, f"Leakage features present: {leak_terms}"
    # Basic shape sanity
    X = np.load(features)
    assert X.shape[1] == len(fn), 'Feature name count mismatch'
