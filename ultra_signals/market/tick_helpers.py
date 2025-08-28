"""Utilities for inferring trade side / signed notional from AggTradeEvent fields.

Keep lightweight: prefer using AggTradeEvent.is_buyer_maker when available. If not provided
users can set use_notional and pass signed-notional into estimator.add_sample directly.
"""
from typing import Optional


def tick_rule_sign(is_buyer_maker: Optional[bool]) -> int:
    """Return +1 for buy-initiated (aggressor buyer), -1 for sell-initiated, or 0 unknown."""
    if is_buyer_maker is None:
        return 0
    # Project convention: is_buyer_maker True means buyer is maker => aggressor is seller
    return -1 if is_buyer_maker else 1


__all__ = ['tick_rule_sign']
