"""Adaptive clustering utilities (Sprint 43).

Attempts HDBSCAN (if installed) else falls back to KMeans with optional
silhouette-based K selection. Provides a uniform interface returning:
- labels: array[int]
- meta: dict with {'method':..., 'params':..., 'n_clusters': int}

Heuristics for mapping clusters to canonical regime labels will live in a
separate helper (placeholder provided).
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import numpy as np

try:
    import hdbscan  # type: ignore
    _HAVE_HDBSCAN = True
except Exception:
    _HAVE_HDBSCAN = False

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

_DEFAULT_LABELS = [
    "trend_up", "trend_down", "chop_lowvol", "panic_deleverage",
    "gamma_pin", "carry_unwind", "risk_on", "risk_off"
]

def adaptive_cluster(X: np.ndarray, settings: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    if X.shape[0] < 10:
        return np.zeros(X.shape[0], dtype=int), {"method":"na","n_clusters":1}
    cfg = settings.get('clusterer', {}) if isinstance(settings, dict) else {}
    method = cfg.get('method','hdbscan')
    if method == 'hdbscan' and _HAVE_HDBSCAN:
        min_cluster_size = int(cfg.get('min_cluster_size',40))
        min_samples = int(cfg.get('min_samples',10))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(X)
        n = len(set(l for l in labels if l!=-1))
        return labels, {"method":"hdbscan","n_clusters":n,"params":{"min_cluster_size":min_cluster_size,"min_samples":min_samples}}
    # Fallback KMeans with silhouette scan
    k = int(cfg.get('k',8))
    best_k = k
    best_score = -1
    best_labels = None
    for cand in range(3, min(12, X.shape[0]-1)):
        try:
            km = KMeans(n_clusters=cand, n_init='auto', random_state=42)
            lbls = km.fit_predict(X)
            if len(set(lbls)) <= 1:
                continue
            score = silhouette_score(X, lbls)
            if score > best_score:
                best_score = score
                best_k = cand
                best_labels = lbls
        except Exception:
            continue
    if best_labels is None:
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
        best_labels = km.fit_predict(X)
        best_k = k
    return best_labels, {"method":"kmeans","n_clusters":int(best_k),"silhouette":float(best_score)}

# --- Cluster → Canonical Regime Label Mapping (placeholder) ---

def map_clusters_to_regimes(X: np.ndarray, labels: np.ndarray, regimes: List[str]) -> Dict[int,str]:
    """Very naive heuristic mapping: order clusters by mean of first feature.
    In practice we will use directional / volatility / macro dimensions.
    """
    unique = [c for c in sorted(set(labels)) if c != -1]
    mapping: Dict[int,str] = {}
    if not unique:
        return mapping
    feat = X[:,0]
    means = {c: float(feat[labels==c].mean()) for c in unique}
    ordered = sorted(unique, key=lambda c: means[c], reverse=True)
    for idx, c in enumerate(ordered):
        mapping[c] = regimes[idx % len(regimes)]
    return mapping

__all__ = ["adaptive_cluster","map_clusters_to_regimes"]
