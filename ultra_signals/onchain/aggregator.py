"""Cohort aggregator: rolling windows, netflows, burst z-scores.

Very small in-memory aggregator intended for unit tests and as a starter
implementation. Uses fixed windows and incremental sums. Not optimized for
high-throughput production but matches acceptance tests (direction/USD/zscore).
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math
import time


def _now_ms() -> int:
    return int(time.time() * 1000)


class CohortAggregator:
    def __init__(self, windows_ms: Dict[str, int] = None):
        # windows: '15m','1h','24h'
        self.windows_ms = windows_ms or {'15m': 15 * 60 * 1000, '1h': 60 * 60 * 1000, '24h': 24 * 60 * 60 * 1000}
        # state structure: symbol -> cohort -> list of (ts_ms, usd, direction)
        self._records: Dict[str, Dict[str, List[Tuple[int, float, str]]]] = {}

    def add_flow(self, symbol: str, cohort: str, direction: str, usd: float, ts_ms: int = None):
        ts = ts_ms or _now_ms()
        sym = self._records.setdefault(symbol, {})
        lst = sym.setdefault(cohort, [])
        lst.append((ts, float(usd), direction))

    def _prune(self, arr: List[Tuple[int, float, str]], cutoff: int) -> List[Tuple[int, float, str]]:
        return [r for r in arr if r[0] >= cutoff]

    def snapshot(self, symbol: str, now_ms: int = None) -> Dict[str, Any]:
        now = now_ms or _now_ms()
        res: Dict[str, Any] = {'symbol': symbol, 'now': now, 'cohorts': {}}
        sym = self._records.get(symbol, {})
        for cohort, arr in sym.items():
            cohort_res = {}
            for name, win_ms in self.windows_ms.items():
                cutoff = now - win_ms
                pr = self._prune(arr, cutoff)
                inflows = sum(a for (_, a, d) in pr if d == 'DEPOSIT')
                outflows = sum(a for (_, a, d) in pr if d == 'WITHDRAWAL')
                net = inflows - outflows
                cohort_res[f'net_{name}'] = net
                cohort_res[f'inflow_{name}'] = inflows
                cohort_res[f'outflow_{name}'] = outflows
            # simple velocity: last value / window seconds
            if arr:
                last_ts, last_amt, last_dir = arr[-1]
                cohort_res['last_flow_usd'] = last_amt
                cohort_res['last_flow_dir'] = last_dir
            res['cohorts'][cohort] = cohort_res
        # compute simple global stats (sum across cohorts)
        global_res = {}
        for name in self.windows_ms.keys():
            global_res[f'net_{name}'] = sum(c.get(f'net_{name}', 0.0) for c in res['cohorts'].values())
            global_res[f'inflow_{name}'] = sum(c.get(f'inflow_{name}', 0.0) for c in res['cohorts'].values())
            global_res[f'outflow_{name}'] = sum(c.get(f'outflow_{name}', 0.0) for c in res['cohorts'].values())
        res['global'] = global_res
        return res


def zscore(value: float, history: List[float]) -> float:
    if not history:
        return 0.0
    mean = sum(history) / len(history)
    var = sum((x - mean) ** 2 for x in history) / len(history)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (value - mean) / std


__all__ = ['CohortAggregator', 'zscore']
