import pandas as pd
import pathlib as pl
import json

p = pl.Path("data/BTCUSDT_5m.csv")
assert p.exists(), f"CSV missing: {p.resolve()}"

# Find a timestamp column
head = pd.read_csv(p, nrows=2000)
lower = {c.lower(): c for c in head.columns}
ts = None
for cand in ("timestamp","time","open_time","date","datetime"):
    if cand in lower:
        ts = lower[cand]
        break
assert ts, f"No timestamp-like column found. Columns: {list(head.columns)}"

# Try ISO, fallback to epoch (s/ms)
def read_full(ts):
    try:
        d = pd.read_csv(p, parse_dates=[ts])
    except Exception:
        v = pd.read_csv(p, usecols=[ts]).iloc[0, 0]
        v = int(v)
        unit = "ms" if v > 10_000_000_000 else "s"
        d = pd.read_csv(p, converters={ts: lambda x: pd.to_datetime(int(x), unit=unit, utc=True)})
        d[ts] = d[ts].dt.tz_convert("UTC").dt.tz_localize(None)
        return d, unit
    else:
        if getattr(d[ts].dt, "tz", None) is not None:
            d[ts] = d[ts].dt.tz_convert("UTC").dt.tz_localize(None)
        return d, None

d, unit = read_full(ts)
d = d.sort_values(ts)
info = {
    "timestamp_col": ts,
    "epoch_unit": unit,
    "rows": int(len(d)),
    "first": str(d[ts].iloc[0]),
    "last": str(d[ts].iloc[-1]),
}

# Check your WF window: 2023-04-04 .. 2023-05-04
start, end = pd.Timestamp("2023-04-04"), pd.Timestamp("2023-05-04")
m = (d[ts] >= start) & (d[ts] < end)
info["slice_rows_2023_04_04__2023_05_04"] = int(m.sum())

print(json.dumps(info, indent=2))
