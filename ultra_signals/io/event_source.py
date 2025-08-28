"""Pluggable event sources: CSV/Parquet reader that normalizes to snapshot/delta/trade events.

Lightweight, no external keys. Files are expected to contain at least timestamp and type-specific fields.
"""
from __future__ import annotations
from typing import Iterator, Dict, Any, Optional
import os
import csv
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # minimal normalization: ensure ts in ms and type
    out = dict(row)
    if 'ts' in out:
        out['ts'] = int(out['ts'])
    elif 'timestamp' in out:
        out['ts'] = int(out['timestamp'])
    return out


class FileEventSource:
    def __init__(self, path: str, format: Optional[str] = None):
        self.path = path
        self.format = format or ("parquet" if path.lower().endswith('.parquet') else 'csv')

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.format == 'csv':
            yield from self._iter_csv()
        elif self.format == 'parquet':
            yield from self._iter_parquet()
        else:
            raise ValueError("unsupported format")

    def _iter_csv(self) -> Iterator[Dict[str, Any]]:
        with open(self.path, 'r', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                yield _normalize_row(row)

    def _iter_parquet(self) -> Iterator[Dict[str, Any]]:
        if pq is None:
            raise RuntimeError("pyarrow not available for parquet reading")
        t = pq.read_table(self.path)
        df = t.to_pandas()
        for _, row in df.iterrows():
            yield _normalize_row(row.to_dict())


class ExchangeAdapter:
    """Base adapter that maps common exchange dump schemas to event format.

    Subclasses should implement map_row(row) -> normalized event dict.
    """
    def map_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()


class BinanceDumpAdapter(ExchangeAdapter):
    def map_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # Support common dumped fields for trades and depth snapshots/deltas
        typ = row.get('type') or row.get('event') or row.get('ev')
        ev = {'ts': int(row.get('ts') or row.get('timestamp') or 0), 'type': typ}
        if typ in ('trade','agg_trade'):
            ev.update({'side': row.get('side') or row.get('s') or row.get('buyer_side'), 'price': float(row.get('price') or row.get('p') or 0), 'size': float(row.get('size') or row.get('q') or 0)})
        elif typ in ('snapshot',):
            # assume bids/asks serialized as string ";" separated px:qty
            bids = row.get('bids')
            asks = row.get('asks')
            if isinstance(bids, str):
                bids = [tuple(map(float, b.split(':'))) for b in bids.split(';') if b]
            if isinstance(asks, str):
                asks = [tuple(map(float, a.split(':'))) for a in asks.split(';') if a]
            ev.update({'data': {'bids': bids or [], 'asks': asks or []}, 'seq': row.get('seq')})
        elif typ in ('delta', 'depthUpdate'):
            bids = row.get('bids')
            asks = row.get('asks')
            if isinstance(bids, str):
                bids = [tuple(map(float, b.split(':'))) for b in bids.split(';') if b]
            if isinstance(asks, str):
                asks = [tuple(map(float, a.split(':'))) for a in asks.split(';') if a]
            ev.update({'data': {'bids': bids or [], 'asks': asks or []}, 'seq': row.get('seq')})
        return ev


class BybitDumpAdapter(ExchangeAdapter):
    def map_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # best effort mapping
        typ = row.get('type') or row.get('event')
        ev = {'ts': int(row.get('ts') or row.get('timestamp') or 0), 'type': typ}
        if typ == 'trade':
            ev.update({'side': row.get('side') or row.get('direction'), 'price': float(row.get('price') or 0), 'size': float(row.get('size') or row.get('qty') or 0)})
        elif typ in ('snapshot','depth'):
            ev.update({'data': {'bids': row.get('bids') or [], 'asks': row.get('asks') or []}, 'seq': row.get('seq')})
        elif typ in ('delta','depthUpdate'):
            ev.update({'data': {'bids': row.get('bids') or [], 'asks': row.get('asks') or []}, 'seq': row.get('seq')})
        return ev


class OKXDumpAdapter(ExchangeAdapter):
    def map_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        typ = row.get('type')
        ev = {'ts': int(row.get('ts') or row.get('timestamp') or 0), 'type': typ}
        if typ == 'trade':
            ev.update({'side': row.get('side'), 'price': float(row.get('px') or row.get('price') or 0), 'size': float(row.get('sz') or row.get('size') or 0)})
        else:
            ev.update({'data': {'bids': row.get('bids') or [], 'asks': row.get('asks') or []}, 'seq': row.get('seq')})
        return ev


__all__ = ["FileEventSource", "ExchangeAdapter", "BinanceDumpAdapter", "BybitDumpAdapter", "OKXDumpAdapter"]


def iter_mapped_events(path: str):
    """Convenience: open file and map rows using exchange adapter guessed from filename.

    Yields normalized event dicts.
    """
    fname = path.lower()
    src = None
    if 'binance' in fname:
        src = BinanceDumpAdapter()
    elif 'bybit' in fname:
        src = BybitDumpAdapter()
    elif 'okx' in fname or 'okex' in fname:
        src = OKXDumpAdapter()
    else:
        src = BinanceDumpAdapter()  # default best-effort

    fes = FileEventSource(path)
    for row in fes:
        try:
            yield src.map_row(row)
        except Exception:
            # fallback: return raw normalized row
            yield row
