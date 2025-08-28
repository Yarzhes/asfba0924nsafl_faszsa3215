"""Book rebuilder utilities: snapshot + delta application with basic QA.

This is an incremental, well-tested building block for the tick-level backtest sprint.
Keep dependency-free and deterministic so unit tests are fast.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional


class RebuildError(Exception):
    pass


class BookRebuilder:
    """Simple L2 book rebuilder with basic integrity checks.

    API:
      - load_snapshot(snapshot)
      - apply_delta(delta)
      - snapshot() -> dict
      - best_bid/best_ask/microprice
    """

    def __init__(self, max_levels: int = 50):
        self.max_levels = int(max_levels)
        # internal representation: price -> size for quick upsert
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}
        self._last_seq = None
        self._resync_needed = False
        # small buffer for out-of-order deltas: seq -> delta
        self._oob_buffer = {}
        # dedupe seen seqs
        self._seen_seqs = set()
        # provenance counters
        self.provenance = {"applied_deltas": 0, "resyncs": 0, "gap_count": 0}

    def load_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Load a full snapshot. Snapshot should contain 'bids' and 'asks' lists of (px, qty).

        Accepts optional 'seq' field for sequencing.
        """
        bids = snapshot.get("bids") or []
        asks = snapshot.get("asks") or []
        seq = snapshot.get("seq")
        self._bids = {float(px): float(qty) for px, qty in bids if float(qty) > 0}
        self._asks = {float(px): float(qty) for px, qty in asks if float(qty) > 0}
        self._last_seq = int(seq) if seq is not None else None
        self._resync_needed = False
        self._trim()
        # clear buffers on fresh snapshot
        self._oob_buffer.clear()
        self._seen_seqs.clear()
        if not self._qa_pass():
            raise RebuildError("Snapshot failed QA: negative spread or invalid sizes")

    def apply_delta(self, delta: Dict[str, Any]) -> None:
        """Apply incremental depth delta.

        Delta format: optional 'seq' integer; 'bids' list of (px, qty) to upsert (qty==0 remove);
        same for 'asks'. Tolerant to out-of-order: if seq is provided and gap detected -> resync flag.
        """
        seq = delta.get("seq")
        src = delta.get("src") or "unknown"
        # Deduplicate
        if seq is not None and seq in self._seen_seqs:
            return
        if seq is not None:
            # simple continuity check
            if self._last_seq is not None and seq <= self._last_seq:
                # older delta, ignore
                return
            if self._last_seq is not None and seq > self._last_seq + 1:
                # gap detected -> store into OOB buffer and mark gap
                self._oob_buffer[int(seq)] = delta
                self._resync_needed = True
                self.provenance["gap_count"] += 1
                return
        # Apply depth updates
        if "bids" in delta:
            for px, qty in delta.get("bids") or []:
                pxf = float(px)
                qf = float(qty)
                if qf <= 0:
                    self._bids.pop(pxf, None)
                else:
                    self._bids[pxf] = qf
        if "asks" in delta:
            for px, qty in delta.get("asks") or []:
                pxf = float(px)
                qf = float(qty)
                if qf <= 0:
                    self._asks.pop(pxf, None)
                else:
                    self._asks[pxf] = qf
        if seq is not None:
            self._last_seq = int(seq)
            self._seen_seqs.add(int(seq))
            self.provenance["applied_deltas"] = self.provenance.get("applied_deltas", 0) + 1
            # attempt to apply any buffered sequential deltas
            self._apply_buffered()
        self._trim()
        if not self._qa_pass():
            # mark for resync; keep best-effort state
            self._resync_needed = True

    def _apply_buffered(self) -> None:
        if self._last_seq is None:
            return
        nxt = self._last_seq + 1
        applied = 0
        while nxt in self._oob_buffer:
            d = self._oob_buffer.pop(nxt)
            # direct recursive apply (no seq re-check)
            for px, qty in d.get("bids") or []:
                pxf = float(px)
                qf = float(qty)
                if qf <= 0:
                    self._bids.pop(pxf, None)
                else:
                    self._bids[pxf] = qf
            for px, qty in d.get("asks") or []:
                pxf = float(px)
                qf = float(qty)
                if qf <= 0:
                    self._asks.pop(pxf, None)
                else:
                    self._asks[pxf] = qf
            self._last_seq = nxt
            self._seen_seqs.add(nxt)
            applied += 1
            nxt += 1
        if applied > 0:
            # successful healing
            self.provenance["resyncs"] = self.provenance.get("resyncs", 0) + 1
            self._resync_needed = False

    def _trim(self) -> None:
        # keep only top N levels
        if len(self._bids) > self.max_levels:
            # keep highest prices
            keep = dict(sorted(self._bids.items(), key=lambda kv: kv[0], reverse=True)[: self.max_levels])
            self._bids = keep
        if len(self._asks) > self.max_levels:
            keep = dict(sorted(self._asks.items(), key=lambda kv: kv[0])[: self.max_levels])
            self._asks = keep

    def _qa_pass(self) -> bool:
        # spread must be non-negative and sizes non-negative
        if not self._bids or not self._asks:
            return True
        best_bid = max(self._bids.keys())
        best_ask = min(self._asks.keys())
        if best_ask < best_bid:
            return False
        for qty in list(self._bids.values()) + list(self._asks.values()):
            if qty < 0:
                return False
        return True

    def needs_resync(self) -> bool:
        return bool(self._resync_needed)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "bids": sorted([(px, self._bids[px]) for px in self._bids], key=lambda x: x[0], reverse=True),
            "asks": sorted([(px, self._asks[px]) for px in self._asks], key=lambda x: x[0]),
            "seq": self._last_seq,
            "resync": self._resync_needed,
        }

    def best_bid(self) -> Optional[float]:
        return max(self._bids.keys()) if self._bids else None

    def best_ask(self) -> Optional[float]:
        return min(self._asks.keys()) if self._asks else None

    def microprice(self) -> Optional[float]:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0


__all__ = ["BookRebuilder", "RebuildError"]
