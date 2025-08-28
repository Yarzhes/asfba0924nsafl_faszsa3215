"""Inspect the FeatureView manifest stored in orderflow_features.db.

Usage:
  Set PYTHONPATH to repo root if needed and run:
    $env:PYTHONPATH='C:\\Users\\Almir\\Projects\\Trading Helper'; python tools/inspect_manifest.py
"""
import json
import sqlite3
import sys

DB = 'orderflow_features.db'
KEY = 'featureview_manifest:sentiment_topics'

def main(db_path: str = DB, key: str = KEY) -> int:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('SELECT v FROM featureview_meta WHERE k=?', (key,))
        r = cur.fetchone()
        if not r:
            print('manifest missing')
            return 1
        obj = json.loads(r[0])
        print(json.dumps(obj, indent=2))
        return 0
    except Exception as e:
        print('error inspecting manifest:', e)
        return 2
    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == '__main__':
    sys.exit(main())
