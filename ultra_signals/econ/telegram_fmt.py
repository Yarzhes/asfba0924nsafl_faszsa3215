"""Telegram formatting helpers for economic events."""
from __future__ import annotations
from typing import Optional
from ultra_signals.core.custom_types import EconomicEvent

def format_pre_alert(ev: EconomicEvent, minutes: int) -> str:
    return (f"[ECON PRE] {ev.title} ({ev.cls.value.upper()} {ev.severity.value.upper()}) in {minutes}m\n"
            f"Risk Window: pre={ev.risk_pre_min}m post={ev.risk_post_min}m")

def format_live(ev: EconomicEvent) -> str:
    return f"[ECON LIVE] {ev.title} {ev.cls.value.upper()} severity={ev.severity.value.upper()}"

def format_done(ev: EconomicEvent) -> str:
    base = f"[ECON DONE] {ev.title} {ev.cls.value.upper()}"
    if ev.surprise_score is not None:
        return base + f" surprise={ev.surprise_score:+.2f}"
    return base
