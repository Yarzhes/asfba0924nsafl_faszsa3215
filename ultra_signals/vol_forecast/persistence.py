from pathlib import Path
import joblib
import json
from typing import Optional, Dict, Any
from datetime import datetime

ROOT = Path("./.vol_models").resolve()
ROOT.mkdir(parents=True, exist_ok=True)


def model_path(symbol: str, timeframe: str) -> Path:
    return ROOT / f"{symbol}__{timeframe}.joblib"


def registry_path() -> Path:
    return ROOT / "registry.json"


def save_model(mgr, symbol: str, timeframe: str):
    p = model_path(symbol, timeframe)
    # store manager (lightweight) as joblib
    joblib.dump(mgr, p)


def load_model(symbol: str, timeframe: str = "default"):
    p = model_path(symbol, timeframe)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def update_registry(symbol: str, timeframe: str, meta: Dict[str, Any], last_refit_ts: datetime):
    rp = registry_path()
    data = {}
    if rp.exists():
        try:
            data = json.loads(rp.read_text())
        except Exception:
            data = {}

    key = f"{symbol}__{timeframe}"
    data[key] = {"meta": meta, "last_refit_ts": last_refit_ts.isoformat()}
    rp.write_text(json.dumps(data, indent=2))


def read_registry() -> Dict[str, Any]:
    rp = registry_path()
    if not rp.exists():
        return {}
    try:
        return json.loads(rp.read_text())
    except Exception:
        return {}
