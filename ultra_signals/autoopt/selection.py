"""Selection logic: rank by median score / stability then tie-breakers."""
from __future__ import annotations
from typing import List, Dict, Any

def rank_candidates(cands: List[Dict[str,Any]], uplift_min: float, baseline_score: float) -> Dict[str,Any]:
    if not cands:
        return {'champion':None,'challengers':[],'promote':False}
    # Sort: score desc, max_dd asc, ret_iqr asc, fees_funding asc
    ordered = sorted(cands, key=lambda x: (-x['score'], x.get('max_dd_pct',9e9), x.get('ret_iqr',9e9), x.get('fees_funding_pct',9e9)))
    champion = ordered[0]
    promote = (champion['score'] - baseline_score) / max(abs(baseline_score),1e-9) >= uplift_min if baseline_score != float('-inf') else champion['score']>0
    challengers = ordered[1:4]
    return {'champion':champion,'challengers':challengers,'promote':promote}

__all__ = ['rank_candidates']
