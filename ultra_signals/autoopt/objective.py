"""Risk-aware objective & constraints for auto-optimization.

Implements scoring formula from Sprint 27 spec. Returns both raw metrics
and final score; applies hard veto constraints mapping to -inf score.
"""
from __future__ import annotations
from typing import Dict, Any
import math

CLAMP = lambda v,lo,hi: max(lo,min(hi,v))

DEFAULT_WEIGHTS = {
    'pf': 1.0,
    'sortino': 0.8,
    'wr': 0.3,
    'dd': 0.8,
    'es': 0.6,
    'turnover': 0.3,
    'fees_funding': 0.2,
}

DEFAULT_CONSTRAINTS = {
    'min_trades': 40,
    'max_dd_pct': 12.0,
    'max_es_pct': 14.0,
    'min_pf': 1.2,
    'min_sortino': 1.0,
    'max_ret_iqr': 0.25,
}

def compute_risk_aware_score(metrics: Dict[str,Any], weights: Dict[str,float]|None=None, constraints: Dict[str,Any]|None=None) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    c = {**DEFAULT_CONSTRAINTS, **(constraints or {})}
    # Constraints check (use medians / distribution keys if provided)
    trades = metrics.get('trades',0)
    if trades < c['min_trades']: return float('-inf')
    if metrics.get('max_dd_pct',0) > c['max_dd_pct']: return float('-inf')
    if metrics.get('cvar_95_pct',0) > c['max_es_pct']: return float('-inf')
    if metrics.get('profit_factor_median', metrics.get('profit_factor',0)) < c['min_pf']: return float('-inf')
    if metrics.get('sortino_median', metrics.get('sortino',0)) < c['min_sortino']: return float('-inf')
    if metrics.get('ret_iqr',0) > c['max_ret_iqr']: return float('-inf')

    pf = CLAMP(metrics.get('profit_factor',0),0,3.0)
    sortino = CLAMP(metrics.get('sortino',0),0,4.0)
    winrate = metrics.get('winrate',0)
    score = (
        w['pf']*pf +
        w['sortino']*sortino +
        w['wr']*winrate
    ) - (
        w['dd']*metrics.get('max_dd_pct',0) +
        w['es']*metrics.get('cvar_95_pct',0) +
        w['turnover']*metrics.get('turnover_penalty',0) +
        w['fees_funding']*metrics.get('fees_funding_pct',0)
    )
    return score

__all__ = ['compute_risk_aware_score','DEFAULT_WEIGHTS','DEFAULT_CONSTRAINTS']
