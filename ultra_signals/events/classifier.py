"""Classifier mapping raw provider events → internal categories & severities.

Heuristics are intentionally simple placeholders so unit tests can validate
mapping behavior. Real implementation can expand rules / ML later.
"""
from __future__ import annotations
from typing import Dict, Any


CATEGORY_KEYWORDS = {
    'CPI': ['cpi'],
    'FOMC': ['fomc', 'federal reserve', 'rate decision'],
    'NFP': ['non-farm', 'nfp'],
    'ECB': ['ecb'],
    'PMI': ['pmi'],
    'PCE': ['pce'],
    'ETF_FLOW': ['etf flow', 'etf net'],
    'EXCH_MAINT': ['maintenance', 'upgrade'],
    'LISTING': ['listing'],
    'AIRDROP': ['airdrop'],
    'PROTOCOL_FORK': ['fork'],
    'FUNDING_SPIKE': ['funding spike'],
}

HIGH_SET = {'CPI','FOMC','NFP','ECB','PROTOCOL_FORK','EXCH_MAINT'}
MED_SET = {'PMI','PCE','ETF_FLOW','LISTING'}


def classify(raw: Dict[str, Any]) -> Dict[str, Any]:
    name = (raw.get('name') or '').lower()
    cat = 'MISC'
    for c, kws in CATEGORY_KEYWORDS.items():
        if any(k in name for k in kws):
            cat = c
            break
    if cat in HIGH_SET:
        imp = 3
    elif cat in MED_SET:
        imp = 2
    else:
        imp = int(raw.get('importance') or 1)
        if imp < 1 or imp > 3:
            imp = 1
    # symbol scope rules
    if cat in {'CPI','FOMC','NFP','ECB','PMI','PCE'}:
        scope = 'GLOBAL'
    elif cat in {'ETF_FLOW'}:
        scope = 'BTCUSDT'  # placeholder; real impl may map specific asset
    else:
        scope = raw.get('symbol') or 'GLOBAL'
    return {
        'id': raw.get('id'),
        'provider': raw.get('provider'),
        'name': raw.get('name'),
        'category': cat,
        'importance': imp,
        'symbol_scope': scope,
        'country': raw.get('country'),
        'start_ts': raw.get('start_ts'),
        'end_ts': raw.get('end_ts'),
        'source_payload': raw.get('payload'),
    }


__all__ = ["classify"]
