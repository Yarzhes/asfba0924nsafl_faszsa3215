"""Streaming orderflow metric calculator (pure functions + small helper class).

The module provides testable helpers for:
 - Cumulative delta (CVD)
 - Order book imbalance
 - Tape metrics (TPS/VPS/NPS) and burst detection
 - Simple footprint absorption heuristic

The implementation is lightweight and import-safe for unit tests.
"""

from __future__ import annotations

import collections
import statistics
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple


def compute_cvd(trades: Sequence[Dict]) -> Dict[str, float]:
    """Compute cumulative delta metrics from a sequence of trades.

    Each trade should be a mapping with keys: `size` (float), `side` ('buy'|'sell').

    Returns dict with cvd_abs, total_volume, cvd_pct.
    """
    cvd = 0.0
    total = 0.0
    for t in trades:
        sz = float(t.get("size", 0.0))
        side = t.get("side", "buy")
        total += abs(sz)
        if side == "buy":
            cvd += sz
        else:
            cvd -= sz
    pct = (cvd / total) if total > 0 else 0.0
    return {"cvd_abs": cvd, "total_volume": total, "cvd_pct": pct}


def compute_ob_imbalance(bids: Sequence[Tuple[float, float]], asks: Sequence[Tuple[float, float]], top_n: int = 5) -> float:
    """Compute order book imbalance over top_n levels.

    bids and asks are lists of (price, size) tuples, sorted by best price first
    (bids descending, asks ascending). The formula is:

      (sum_bid - sum_ask) / (sum_bid + sum_ask)

    Returns 0.0 if there is no liquidity at the selected depth.
    """
    bid_sum = sum(sz for _, sz in list(bids)[:top_n])
    ask_sum = sum(sz for _, sz in list(asks)[:top_n])
    denom = bid_sum + ask_sum
    if denom == 0:
        return 0.0
    return (bid_sum - ask_sum) / denom


def compute_tape_metrics(trades: Sequence[Dict], window_s: float) -> Dict[str, float]:
    """Compute TPS, VPS, NPS over the provided trades assumed to fall inside window_s seconds.

    Each trade should have `size` and `price`.
    """
    n = len(trades)
    total_size = sum(float(t.get("size", 0.0)) for t in trades)
    total_notional = sum(float(t.get("size", 0.0)) * float(t.get("price", 0.0)) for t in trades)
    tps = n / window_s if window_s > 0 else 0.0
    vps = total_size / window_s if window_s > 0 else 0.0
    nps = total_notional / window_s if window_s > 0 else 0.0
    return {"tape_tps": tps, "tape_vps": vps, "tape_nps": nps}


def detect_vps_burst(current_vps: float, history_vps: Sequence[float], sigma: float = 2.0) -> bool:
    """Detect a burst when current_vps > mean(history) + sigma * std(history).

    If history is empty or std==0, the function falls back to comparing against mean only.
    """
    if not history_vps:
        return False
    mean = statistics.mean(history_vps)
    std = statistics.pstdev(history_vps) if len(history_vps) >= 1 else 0.0
    threshold = mean + sigma * std
    return current_vps > threshold


def detect_absorption(trade: Dict, pre_level_size: float, post_level_size: float, min_cluster_volume: float = 0.0) -> bool:
    """Simple footprint absorption heuristic.

    Returns True when a large aggressive trade at a price is met with equal-or-higher
    passive liquidity at that price after execution (i.e., liquidity added / not removed).

    This heuristic is intentionally conservative and lightweight — a true footprint
    detector would need per-price-time aggregation and book diffs.
    """
    sz = float(trade.get("size", 0.0))
    if sz < float(min_cluster_volume):
        return False
    # If post-level size is greater or equal to pre-level, we interpret that as absorption
    return post_level_size >= pre_level_size


class OrderflowCalculator:
    """Lightweight streaming calculator that keeps short sliding windows.

    The class is intentionally small: it stores recent trades and VPS history and
    can compute the standard metrics on demand. It does not perform IO by default
    but accepts a writer to persist computed FeatureView records.
    """

    def __init__(self, vps_window_s: int = 10, cvd_window_s: int = 60, history_len: int = 100):
        self.vps_window_s = vps_window_s
        self.cvd_window_s = cvd_window_s
        self.trades: Deque[Dict] = collections.deque()
        # circular buffer of past VPS values for burst detection
        self.vps_history: Deque[float] = collections.deque(maxlen=history_len)
        self.last_book: Optional[Dict[str, List[Tuple[float, float]]]] = None

    def ingest_trade(self, trade: Dict) -> None:
        """Add a trade to the streaming window. Trade must contain `ts` (seconds), `size`, `price`, `side`.
        """
        self.trades.append(trade)
        # prune trades older than max window (use cvd and vps windows)
        now = float(trade.get("ts", time.time()))
        cutoff = now - max(self.vps_window_s, self.cvd_window_s)
        while self.trades and float(self.trades[0].get("ts", now)) < cutoff:
            self.trades.popleft()

    def ingest_book_snapshot(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> None:
        self.last_book = {"bids": bids, "asks": asks}

    def compute_current(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = now or time.time()
        # select trades in cvd window
        cvd_window_cutoff = now - self.cvd_window_s
        cvd_trades = [t for t in self.trades if float(t.get("ts", now)) >= cvd_window_cutoff]
        cvd = compute_cvd(cvd_trades)

        # select trades in vps window
        vps_window_cutoff = now - self.vps_window_s
        vps_trades = [t for t in self.trades if float(t.get("ts", now)) >= vps_window_cutoff]
        tape = compute_tape_metrics(vps_trades, self.vps_window_s)

        # update vps history
        self.vps_history.append(tape["tape_vps"])

        # detect burst
        burst = detect_vps_burst(tape["tape_vps"], list(self.vps_history)[:-1])

        # compute OB imbalance from last snapshot if present
        ob_top1 = ob_top5 = ob_full = 0.0
        if self.last_book:
            ob_top1 = compute_ob_imbalance(self.last_book["bids"], self.last_book["asks"], top_n=1)
            ob_top5 = compute_ob_imbalance(self.last_book["bids"], self.last_book["asks"], top_n=5)
            ob_full = compute_ob_imbalance(self.last_book["bids"], self.last_book["asks"], top_n=50)

        return {
            "cvd_abs": cvd["cvd_abs"],
            "cvd_pct": cvd["cvd_pct"],
            "total_volume": cvd["total_volume"],
            "ob_imbalance_top1": ob_top1,
            "ob_imbalance_top5": ob_top5,
            "ob_imbalance_full": ob_full,
            "tape_tps": tape["tape_tps"],
            "tape_vps": tape["tape_vps"],
            "tape_nps": tape["tape_nps"],
            "tape_burst_flag": 1 if burst else 0,
        }
"""Simple orderflow engine that ingests trades and L2 snapshots/deltas
and produces CVD, orderbook imbalance, tape speed, and lightweight
footprint S/R clusters.

This is intentionally small and pure-python to keep tests fast.
"""
from collections import deque, defaultdict
import math
import time

class OrderflowEngine:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.depth_levels = cfg.get("depth_levels", [1,5,25])
        self.cvd_window = cfg.get("cvd_window", 300)  # seconds
        self.tape_window = cfg.get("tape_window", 10)  # seconds
        self.tape_z_window = cfg.get("tape_z_window", 60)
        self.footprint_min_volume = cfg.get("footprint_min_volume", 1000)

        # internal state
        self.trades = deque()  # (ts, price, size, side)
        self.cvd = 0.0
        self.cvd_history = deque()  # (ts, cvd_delta, volume)

        # orderbook: dict price -> {bid:qty, ask:qty}
        self.orderbook = {"bids":[], "asks":[]}

        # tape windows
        self.tape_events = deque()  # (ts, size, notional, side)

        # footprint clusters: price -> accumulated aggressive volume
        self.footprint = defaultdict(float)

    def ingest_trade(self, ts, price, size, side, aggressor=True):
        """Ingest a trade. side: 'buy'|'sell' as trade taker side. aggressor
        indicates whether trade was aggressive (taker) or passive.
        """
        self.trades.append((ts, price, size, side))
        self.tape_events.append((ts, size, size * price, side))

        # CVD update only counts aggressive trades
        if aggressor:
            delta = size if side == "buy" else -size
            self.cvd += delta
            self.cvd_history.append((ts, delta, size))
            # footprint accumulation: mark price level
            self.footprint[price] += abs(delta)

        self._prune_history(ts)

    def ingest_orderbook_snapshot(self, bids, asks):
        """bids/asks: list of (price, qty), top-first. We'll store shallow copy."""
        self.orderbook["bids"] = list(bids)
        self.orderbook["asks"] = list(asks)

    def ingest_orderbook_delta(self, bids=None, asks=None):
        """Apply simple delta: replace top levels present in lists."""
        if bids:
            for i, (p,q) in enumerate(bids):
                if i < len(self.orderbook["bids"]):
                    self.orderbook["bids"][i] = (p,q)
                else:
                    self.orderbook["bids"].append((p,q))
        if asks:
            for i, (p,q) in enumerate(asks):
                if i < len(self.orderbook["asks"]):
                    self.orderbook["asks"][i] = (p,q)
                else:
                    self.orderbook["asks"].append((p,q))

    def _prune_history(self, now_ts):
        # prune CVD history by cvd_window
        cutoff = now_ts - self.cvd_window
        while self.cvd_history and self.cvd_history[0][0] < cutoff:
            self.cvd_history.popleft()

        # prune tape events by tape_z_window
        cutoff2 = now_ts - self.tape_z_window
        while self.tape_events and self.tape_events[0][0] < cutoff2:
            self.tape_events.popleft()

    def get_cvd(self):
        """Return current cvd_abs, cvd_pct (wrt recent volume), cvd_z (z-score).
        cvd_pct uses rolling sum of absolute traded volume in cvd_window.
        cvd_z uses mean/std over cvd_history of deltas.
        """
        total_vol = sum(abs(v[2]) for v in self.cvd_history) or 1.0
        deltas = [v[1] for v in self.cvd_history]
        mean = sum(deltas) / len(deltas) if deltas else 0.0
        std = math.sqrt(sum((d-mean)**2 for d in deltas)/len(deltas)) if deltas else 0.0
        cvd_z = (self.cvd - mean) / (std or 1.0)
        return {
            "cvd_abs": self.cvd,
            "cvd_pct": self.cvd / total_vol,
            "cvd_z": cvd_z,
        }

    def get_orderbook_imbalance(self, topN=5):
        """Compute OB imbalance for topN levels.
        returns (imbalance, bid_sum, ask_sum)
        """
        bids = self.orderbook.get("bids", [])[:topN]
        asks = self.orderbook.get("asks", [])[:topN]
        bid_sum = sum(q for _, q in bids)
        ask_sum = sum(q for _, q in asks)
        total = bid_sum + ask_sum or 1.0
        imbalance = (bid_sum - ask_sum) / total
        return imbalance, bid_sum, ask_sum

    def get_tape_metrics(self, now_ts=None):
        now_ts = now_ts or time.time()
        window = self.tape_window
        cutoff = now_ts - window
        events = [e for e in self.tape_events if e[0] >= cutoff]
        tps = len(events) / (window or 1.0)
        vps = sum(e[1] for e in events) / (window or 1.0)
        nps = sum(e[2] for e in events) / (window or 1.0)

        # compute z against historical tape_events in tape_z_window
        all_sizes = [e[1] for e in self.tape_events]
        mean = sum(all_sizes)/len(all_sizes) if all_sizes else 0.0
        std = math.sqrt(sum((s-mean)**2 for s in all_sizes)/len(all_sizes)) if all_sizes else 0.0
        burst_flag = (vps > mean + 2*std) if std else False

        return {"tps": tps, "vps": vps, "nps": nps, "burst": burst_flag}

    def detect_footprint_levels(self, top_k=5):
        """Return top_k price levels by absorbed aggressive volume above threshold."""
        items = [(px, vol) for px, vol in self.footprint.items() if vol >= self.footprint_min_volume]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_k]

    def compute_micro_score(self):
        """Simple fused score from components in [-1,1] where positive is bullish."""
        cvd = self.get_cvd()
        imb, _, _ = self.get_orderbook_imbalance(topN=5)
        tape = self.get_tape_metrics()

        # normalize components
        cvd_norm = max(-1.0, min(1.0, cvd["cvd_pct"]))
        imb_norm = max(-1.0, min(1.0, imb))
        tape_norm = 1.0 if tape["burst"] and tape["vps"] > 0 else 0.0

        score = 0.5 * cvd_norm + 0.3 * imb_norm + 0.2 * tape_norm
        return {"of_micro_score": score, "components": {"cvd": cvd_norm, "imb": imb_norm, "tape": tape_norm}}
