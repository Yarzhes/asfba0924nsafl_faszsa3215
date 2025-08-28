"""Arbitrage package exports.

This module re-exports the implemented arbitrage components:
  - data models (snapshots)
  - the `ArbitrageCollector` orchestration layer
  - the `ArbitrageAnalyzer` feature calculations and flags
  - Telegram formatting helper

The heavy lifting lives in `collector.py`, `analyzer.py`, `models.py` and
`telegram.py` so tests and callers can import from this package root.
"""

from .models import (
    VenueQuote,
    VenueDepthSummary,
    ExecutableSpread,
    FundingSnapshot,
    BasisSnapshot,
    GeoPremiumSnapshot,
    ArbitrageFeatureSet,
    OpportunityFlag,
)
from .collector import ArbitrageCollector
from .analyzer import ArbitrageAnalyzer
from .telegram import format_arb_telegram_line

__all__ = [
    'VenueQuote', 'VenueDepthSummary', 'ExecutableSpread', 'FundingSnapshot', 'BasisSnapshot',
    'GeoPremiumSnapshot', 'ArbitrageFeatureSet', 'OpportunityFlag', 'ArbitrageCollector',
    'ArbitrageAnalyzer', 'format_arb_telegram_line'
]
