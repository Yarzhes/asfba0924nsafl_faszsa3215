"""Small diagnostics utility to print per-topic timelines and divergence hit-rate.

Produces simple CSVs under a provided output directory.
"""
import csv
from pathlib import Path
from typing import Dict, Any


def dump_topic_timelines(aggregator, out_dir: str = "./diagnostics"):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    rows = []
    for sym, snap in aggregator.latest_per_symbol.items():
        row = {"symbol": sym, **snap}
        rows.append(row)
    if not rows:
        print("No data in aggregator.latest_per_symbol")
        return
    keys = sorted(rows[0].keys())
    with (p/"topic_timelines.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {p/'topic_timelines.csv'}")


def divergence_hit_rate(aggregator) -> Dict[str, float]:
    total = 0
    hits = 0
    for sym, snap in aggregator.latest_per_symbol.items():
        total += 1
        flags = snap.get("contrarian_flag_long") or snap.get("contrarian_flag_short") or 0
        if flags:
            hits += 1
    if total == 0:
        return {"total": 0, "hits": 0, "hit_rate": 0.0}
    return {"total": total, "hits": hits, "hit_rate": hits/total}
