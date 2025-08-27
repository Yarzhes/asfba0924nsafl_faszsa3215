"""Unified Sentiment & Social Signal Layer (Sprint 40)

This package provides a pluggable pipeline that ingests free/no-key social &
market crowd positioning data, converts it into numeric sentiment features,
computes rolling aggregates, detects extremes, and exposes:
  - Per-symbol sentiment scores (short / medium horizon)
  - Influencer-weighted sentiment
  - Burst / velocity metrics (posts per minute, unique authors)
  - Extreme flags (over-bullish / over-bearish) for optional veto & size dampen
  - Auxiliary features: fear & greed index, funding / OI z-scores, news bursts

Design Goals:
  * Free sources only (Twitter via snscrape, Reddit JSON, alt.me FGI, public
    funding endpoints, optional Google Trends / GDELT).
  * Resilient: caching, polite backoff, source toggles, graceful degradation.
  * Lightweight scoring: lexicon + emoji + optional transformer refinement.
  * Deterministic & testable: pure functions for scoring + aggregation.

Public Entry Points:
  - SentimentEngine (high-level orchestrator)
  - BaseCollector subclasses for each source
  - score_text() and SentimentScorer for local polarity scoring
  - Aggregator for rolling windows & extreme detection

Minimal placeholder implementations are included; extend collectors incrementally.
"""

from .engine import SentimentEngine  # noqa: F401
