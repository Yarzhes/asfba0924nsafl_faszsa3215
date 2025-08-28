"""Register a local FeatureView manifest into a FeatureViewWriter-backed DB.

Usage: run from repo root; specify sqlite path if not default.
This writes the YAML manifest content under key `featureview_manifest:sentiment_topics`
so that downstream components can discover emitted keys.
"""
import sys
import yaml
from ultra_signals.orderflow.persistence import FeatureViewWriter
from pathlib import Path

DB_DEFAULT = "orderflow_features.db"
MANIFEST_PATH = Path("sentiment_topics/feature_manifest.yaml")


def main(sqlite_path: str = DB_DEFAULT):
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}")
        return 1
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    fw = FeatureViewWriter(sqlite_path=sqlite_path)
    fw.set_meta("featureview_manifest:sentiment_topics", manifest)
    print(f"Registered manifest into {sqlite_path} under key featureview_manifest:sentiment_topics")
    fw.close()
    return 0


if __name__ == "__main__":
    sqlite = sys.argv[1] if len(sys.argv) > 1 else DB_DEFAULT
    raise SystemExit(main(sqlite))
