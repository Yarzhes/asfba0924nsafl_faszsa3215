"""Directional Change (DC) engine package.

Lightweight, streaming-friendly sampler and utilities for Directional Change
event detection. This is an initial implementation: percent-based thresholds,
per-threshold state, overshoot tracking, simple event-bar builder and fusion.
"""
from .events import EventType, DCEvent, OSEvent
from .sampler import DirectionalChangeSampler
from .event_bars import EventBarBuilder
from .fusion import Fusion

__all__ = [
    "EventType",
    "DCEvent",
    "OSEvent",
    "DirectionalChangeSampler",
    "EventBarBuilder",
    "Fusion",
]
