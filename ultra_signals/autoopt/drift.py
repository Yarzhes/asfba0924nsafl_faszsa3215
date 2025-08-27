"""Simple drift detection utilities (feature distribution shift).

We place lightweight Population Stability Index (PSI) and KL divergence
helpers; heavy logic can be added later. Currently used for optional
logging / size reduction triggers.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence

EPS=1e-9

def psi(expected: Sequence[float], actual: Sequence[float], bins: int=10) -> float:
    if len(expected)==0 or len(actual)==0:
        return 0.0
    e = np.asarray(expected); a = np.asarray(actual)
    qs = np.linspace(0,1,bins+1)
    cuts = np.quantile(e, qs)
    cuts[0]-=1e-9; cuts[-1]+=1e-9
    e_counts,_ = np.histogram(e, bins=cuts)
    a_counts,_ = np.histogram(a, bins=cuts)
    e_rat = e_counts / max(e_counts.sum(),1)
    a_rat = a_counts / max(a_counts.sum(),1)
    val = 0.0
    for er,ar in zip(e_rat,a_rat):
        if er<EPS and ar<EPS: continue
        val += (ar-er)*np.log((ar+EPS)/(er+EPS))
    return float(abs(val))

def kl_div(p: Sequence[float], q: Sequence[float], bins: int=20) -> float:
    if len(p)==0 or len(q)==0: return 0.0
    p = np.asarray(p); q = np.asarray(q)
    lo = min(p.min(), q.min()); hi = max(p.max(), q.max())
    hist_p,_ = np.histogram(p, bins=bins, range=(lo,hi), density=True)
    hist_q,_ = np.histogram(q, bins=bins, range=(lo,hi), density=True)
    hist_p = hist_p + EPS; hist_q = hist_q + EPS
    hist_p /= hist_p.sum(); hist_q/=hist_q.sum()
    return float(np.sum(hist_p * np.log(hist_p / hist_q)))

__all__ = ['psi','kl_div']
