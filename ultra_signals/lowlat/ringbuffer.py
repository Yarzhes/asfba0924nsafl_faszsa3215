"""Small ring buffer used for single-writer, multiple-reader patterns.

This is an intentionally tiny, dependency-free structure suitable for
in-process synthetic tests. It is not a true lock-free C extension, but
the API is tailored for a single-writer scenario and bounded memory use.
"""
from __future__ import annotations
from typing import List, Optional


class RingBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._cap = capacity
        self._buf: List[Optional[object]] = [None] * capacity
        self._write_idx = 0
        self._read_idx = 0
        self._size = 0

    def push(self, item: object) -> bool:
        """Push an item. Returns False if buffer full (caller may drop)."""
        if self._size >= self._cap:
            return False
        self._buf[self._write_idx] = item
        self._write_idx = (self._write_idx + 1) % self._cap
        self._size += 1
        return True

    def pop(self) -> Optional[object]:
        if self._size == 0:
            return None
        v = self._buf[self._read_idx]
        self._buf[self._read_idx] = None
        self._read_idx = (self._read_idx + 1) % self._cap
        self._size -= 1
        return v

    def peek_all(self) -> List[object]:
        """Return a shallow copy of current items (consumes nothing)."""
        out: List[object] = []
        idx = self._read_idx
        for _ in range(self._size):
            out.append(self._buf[idx])
            idx = (idx + 1) % self._cap
        return out

    def __len__(self) -> int:
        return self._size

    def capacity(self) -> int:
        return self._cap


__all__ = ["RingBuffer"]
