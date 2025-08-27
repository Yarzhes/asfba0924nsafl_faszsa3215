"""Auto-Optimization (Sprint 27) package.

Provides nightly walk-forward tuning with risk-aware selection and safe promotion.
Initial scaffold; implementations kept lightweight to avoid breaking existing code.
"""
__all__ = [
    'spaces','search','objective','wf_runner','selection','publisher','drift'
]
