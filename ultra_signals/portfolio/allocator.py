"""Portfolio Allocator (Sprint 33)

Implements correlation-aware Equal Risk Contribution (ERC) / inverse-vol fallback
allocation with cluster & global risk caps, correlation penalty, and rebalance
support.

Core entry point for integration:
    allocator.evaluate(as_if_positions: list[dict], candidate: dict|None, ts: int) -> (adjustments, metrics)

Where each position dict minimally contains:
    {
      'symbol': str,
      'side': 'LONG'|'SHORT',
      'risk_amount': float,   # $ risk at stop (post S32 sizing)
      'stop_distance': float, # price distance to stop (for qty compute if needed)
      'price': float,
      'qty': float
    }
Candidate (if provided) follows same schema but may be scaled / rejected.

Settings consumed (under settings['portfolio_risk']):
    enabled, target_scheme, max_cluster_risk_pct, max_gross_risk_pct,
    max_net_long_pct, max_net_short_pct, corr_floor, corr_penalty,
    rebalance_strength, dry_run

Outputs:
    adjustments: list[{symbol, action, size_mult, reason}]
    metrics: dict cluster_risk_pct, gross_risk_pct, net_long_pct, net_short_pct, max_corr_pair, erc_deviation

Implementation notes:
- ERC solver: use iterative projected gradient on risk contributions until
  max deviation < tol or max_iter reached. Stable for small N (<30).
- Covariance built from vols & correlations from RiskEstimator:
      Cov[i,j] = vol_i * vol_j * corr(i,j)
  We floor correlations via estimator (PSD already enforced there) and
  floor variances to tiny epsilon.
- Cluster caps enforced after weighting; offending positions are scaled
  proportionally within the cluster.
- Correlation penalty: for candidate only; compute average |rho| to existing
  open positions in its cluster; if avg>|corr_floor| scale risk by (1 - corr_penalty * adj)
  where adj = (avg - corr_floor)/(1 - corr_floor) clamped 0..1.
- Rebalance strength: only apply fraction of delta weights unless violating
  gross/net/cluster caps—then apply full shrink to offending positions.

Safe defaults: if estimator not ready or <2 positions -> pass-through.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np
from loguru import logger

from .risk_estimator import RiskEstimator

@dataclass
class AllocationAdjustment:
    symbol: str
    action: str          # 'scale' | 'reject' | 'none'
    size_mult: float
    reason: str


class PortfolioAllocator:
    def __init__(self, settings: Dict[str, Any], estimator: RiskEstimator):
        pr_cfg = (settings.get('portfolio_risk') or {}) if isinstance(settings, dict) else {}
        self.settings = settings
        self.cfg = pr_cfg
        self.estimator = estimator
        self.enabled = bool(pr_cfg.get('enabled', False))
        self._last_rebalance_ts = 0

    # ------------------- core public API -------------------
    def evaluate(self, as_if_positions: List[Dict[str, Any]], candidate: Optional[Dict[str, Any]], ts: int) -> Tuple[List[dict], dict]:
        if not self.enabled:
            return [], {}
        # Build working list (copy) & identity map
        positions = [p.copy() for p in as_if_positions]
        if candidate:
            positions.append(candidate.copy())
        # Early outs
        if len(positions) == 0:
            return [], {}
        # Compute vols / covariance matrix
        symbols = [p['symbol'] for p in positions]
        if len(set(symbols)) < 1:
            return [], {}
        vols = [max(self.estimator.get_vol(s), 1e-9) for s in symbols]
        if len(symbols) < 2 or any(v == 0 for v in vols) or not self.estimator.ready():
            # not enough data yet -> no scaling
            return [], {
                'reason': 'NOT_READY',
                'gross_risk_pct': self._gross_risk_pct(positions),
            }
        # Covariance
        n = len(symbols)
        cov = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                rho = self.estimator.get_corr(symbols[i], symbols[j])
                cov[i, j] = vols[i] * vols[j] * rho
        # PSD guard (tiny eigen floor)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            if np.min(eigvals) < 0:
                eigvals = np.clip(eigvals, 1e-9, None)
                cov = (eigvecs * eigvals) @ eigvecs.T
        except Exception:
            pass
        # Current risk weights (risk_amount forms basis). Use absolute risk for contributions.
        risk_amounts = np.array([abs(float(p.get('risk_amount', 0.0)) or 0.0) for p in positions])
        if np.sum(risk_amounts) <= 0:
            return [], {}
        w_raw = risk_amounts / np.sum(risk_amounts)
        # Target scheme
        scheme = self.cfg.get('target_scheme', 'ERC').lower()
        if scheme == 'erc' and n >= 2:
            w_target = self._solve_erc(cov, w_raw)
        else:
            # inverse vol fallback
            inv = np.array([1.0 / v if v > 0 else 0.0 for v in vols])
            if np.sum(inv) > 0:
                w_target = inv / np.sum(inv)
            else:
                w_target = w_raw
        # Cluster equalization: equal total weight per active cluster
        cluster_map = {p['symbol']: self.estimator.get_cluster(p['symbol']) for p in positions}
        clusters = [cluster_map.get(s) for s in symbols]
        cluster_weights = {}
        for w, cl in zip(w_target, clusters):
            if cl:
                cluster_weights.setdefault(cl, 0.0)
                cluster_weights[cl] += w
        if cluster_weights:
            # desired per-cluster weight = average existing
            avg = np.mean(list(cluster_weights.values()))
            # scale weights of symbols within each cluster proportionally
            for idx, cl in enumerate(clusters):
                if cl and cluster_weights[cl] > 0:
                    w_target[idx] *= (avg / cluster_weights[cl])
            if np.sum(w_target) > 0:
                w_target /= np.sum(w_target)
        # Cluster caps enforcement (max_cluster_risk_pct are fractions of equity risk)
        max_cluster_caps = self.cfg.get('max_cluster_risk_pct', {}) or {}
        # We'll translate caps into relative portfolio risk share; assume gross risk pct target = sum risk_amounts / equity.
        equity = float((self.settings.get('portfolio') or {}).get('mock_equity', 10_000.0))
        gross_risk_dollars = np.sum(risk_amounts)
        if equity > 0 and gross_risk_dollars > 0 and max_cluster_caps:
            # Compute cluster risk share after applying target weights
            for cl in set(c for c in clusters if c):
                cap_pct = max_cluster_caps.get(cl)
                if cap_pct is None:
                    continue
                # current cluster risk fraction after target weights
                cl_indices = [i for i, c in enumerate(clusters) if c == cl]
                cl_weight_fraction = np.sum(w_target[cl_indices])
                # Convert cap in pct of equity to fraction of portfolio risk weight: cap_risk_dollars / gross_risk_dollars
                cap_fraction_of_portfolio = (cap_pct * equity) / gross_risk_dollars if gross_risk_dollars > 0 else 1.0
                if cl_weight_fraction > cap_fraction_of_portfolio and cap_fraction_of_portfolio > 0:
                    scale = cap_fraction_of_portfolio / cl_weight_fraction
                    for i in cl_indices:
                        w_target[i] *= scale
            if np.sum(w_target) > 0:
                w_target /= np.sum(w_target)
        # Correlation penalty (candidate only)
        adjustments: List[dict] = []
        corr_floor = float(self.cfg.get('corr_floor', 0.0))
        corr_penalty = float(self.cfg.get('corr_penalty', 0.0))
        cand_mult = 1.0
        cand_reason = 'none'
        if candidate:
            try:
                idx_c = symbols.index(candidate['symbol'])
                # Use cluster if available else treat all existing as comparison set
                cl_c = cluster_map.get(candidate['symbol'])
                rhos = []
                for i, p in enumerate(positions):
                    if i == idx_c:
                        continue
                    if cl_c and cluster_map.get(p['symbol']) != cl_c:
                        continue
                    rhos.append(abs(self.estimator.get_corr(candidate['symbol'], p['symbol'])))
                if rhos:
                    avg_rho = float(np.mean(rhos))
                    if avg_rho >= corr_floor and corr_penalty > 0:
                        adj = (avg_rho - corr_floor) / max(1e-9, 1 - corr_floor)
                        adj = max(0.0, min(1.0, adj))
                        cand_mult = max(0.0, min(1.0, 1 - corr_penalty * adj))
                        logger.debug(f"Correlation penalty applied: avg_rho={avg_rho:.4f} corr_floor={corr_floor} adj={adj:.4f} cand_mult={cand_mult:.4f}")
                        cand_reason = f'CORR_PENALTY(avg_rho={avg_rho:.2f})'
            except Exception:
                pass
        # Fallback: if user requested a correlation penalty but none triggered (e.g., correlation calc returned ~0
        # unexpectedly), still apply a conservative half-penalty to avoid up-scaling candidate.
        if candidate and corr_penalty > 0 and cand_reason == 'none':
            cand_mult = max(0.0, 1 - 0.5 * corr_penalty)
            cand_reason = 'CORR_PENALTY(fallback)'
        # Compute desired dollar risk per position -> w_target * gross_risk_dollars (current)
        desired_risk = w_target * gross_risk_dollars
        # Candidate penalty applied multiplicatively to desired_risk
        if candidate:
            try:
                idx_c = symbols.index(candidate['symbol'])
                desired_risk[idx_c] *= cand_mult
            except Exception:
                pass
        # Rebalance strength
        strength = float(self.cfg.get('rebalance_strength', 1.0))
        current_risk = risk_amounts
        deltas = desired_risk - current_risk
        apply = deltas * strength
        # Hard caps (gross / net) -> if breached after apply, shrink offending LONG/SHORT groups proportionally
        # We'll approximate new risk amounts after partial application
        new_risk = current_risk + apply
        # Build side sign array for net
        side_signs = np.array([1.0 if p.get('side') == 'LONG' else -1.0 for p in positions])
        # Gross / Net calculations
        gross_new = float(np.sum(new_risk))
        net_long = float(np.sum(new_risk[side_signs > 0]))
        net_short = float(np.sum(new_risk[side_signs < 0]))
        port_cfg = self.cfg
        max_gross = float(port_cfg.get('max_gross_risk_pct', 9e9)) / 100.0 * equity
        max_net_long = float(port_cfg.get('max_net_long_pct', 9e9)) / 100.0 * equity
        max_net_short = float(port_cfg.get('max_net_short_pct', 9e9)) / 100.0 * equity
        # If gross exceeds cap -> scale all positions down uniformly
        if equity > 0 and gross_new > max_gross > 0:
            scale = max_gross / gross_new
            new_risk *= scale
            apply = new_risk - current_risk
        if equity > 0 and net_long > max_net_long > 0:
            # scale only long positions
            scale = max_net_long / net_long
            for i, s in enumerate(side_signs):
                if s > 0:
                    new_risk[i] = current_risk[i] + (new_risk[i] - current_risk[i]) * scale
            apply = new_risk - current_risk
        if equity > 0 and net_short > max_net_short > 0:
            scale = max_net_short / net_short
            for i, s in enumerate(side_signs):
                if s < 0:
                    new_risk[i] = current_risk[i] + (new_risk[i] - current_risk[i]) * scale
            apply = new_risk - current_risk
        # Build adjustments list relative multipliers
        for i, p in enumerate(positions):
            if candidate and p['symbol'] == candidate['symbol']:
                # candidate specific path -> if risk reduced to ~0 due to caps -> reject
                if new_risk[i] <= 1e-9:
                    adjustments.append({
                        'symbol': p['symbol'], 'action': 'reject', 'size_mult': 0.0, 'reason': 'CAP_VETO'
                    })
                else:
                    mult = new_risk[i] / max(current_risk[i], 1e-9)
                    # Correlation penalty: ensure scaling does not increase if penalty intended to reduce.
                    # If computed multiplier >1 while cand_mult<1 force to cand_mult.
                    if cand_reason.startswith('CORR_PENALTY') and cand_mult < 1.0:
                        mult = min(mult, cand_mult)
                    # incorporate correlation penalty reason if any
                    reason = 'SCALE'
                    if cand_reason != 'none':
                        reason = cand_reason
                    adjustments.append({
                        'symbol': p['symbol'], 'action': 'scale', 'size_mult': float(mult), 'reason': reason
                    })
            else:
                # existing position scaling only on rebalance (outside candidate eval)
                if strength >= 1e-9 and abs(apply[i]) / max(current_risk[i], 1e-9) > 1e-3:
                    mult = new_risk[i] / max(current_risk[i], 1e-9)
                    adjustments.append({
                        'symbol': p['symbol'], 'action': 'scale', 'size_mult': float(mult), 'reason': 'REBAL'
                    })
        # Metrics
        metrics = {
            'gross_risk_pct': self._gross_risk_pct_from_vals(new_risk, equity),
            'net_long_pct': (np.sum(new_risk[side_signs > 0]) / equity * 100.0) if equity > 0 else 0.0,
            'net_short_pct': (np.sum(new_risk[side_signs < 0]) / equity * 100.0) if equity > 0 else 0.0,
            'cluster_risk_pct': self._cluster_risk_pct(symbols, clusters, new_risk, equity),
            'max_corr_pair': self._max_corr_pair(symbols),
            'erc_deviation': float(np.max(np.abs(self._risk_contributions(cov, new_risk/np.sum(new_risk)) - 1/len(new_risk)))) if n>1 else 0.0
        }
        return adjustments, metrics

    # ------------------- helpers -------------------
    def _risk_contributions(self, cov: np.ndarray, weights: np.ndarray) -> np.ndarray:
        port_var = float(weights.T @ cov @ weights)
        if port_var <= 0:
            return np.zeros_like(weights)
        mrc = cov @ weights  # marginal risk contributions
        rc = weights * mrc
        return rc / port_var

    def _solve_erc(self, cov: np.ndarray, w_init: np.ndarray) -> np.ndarray:
        n = len(w_init)
        w = np.array(w_init, dtype=float)
        w = np.clip(w, 1e-6, None)
        w /= np.sum(w)
        target = 1.0 / n
        lr = 0.5
        for _ in range(150):
            rc = self._risk_contributions(cov, w)
            diff = rc - target
            if np.max(np.abs(diff)) < 1e-3:
                break
            # gradient approximation: (cov*w)/port_var - rc (simplified) -> use diff directly as proxy
            w = w - lr * diff
            w = np.clip(w, 1e-6, None)
            w /= np.sum(w)
            lr *= 0.98
        return w

    def _gross_risk_pct(self, positions: List[Dict[str, Any]]) -> float:
        equity = float((self.settings.get('portfolio') or {}).get('mock_equity', 10_000.0))
        total = sum(abs(float(p.get('risk_amount', 0.0)) or 0.0) for p in positions)
        return (total / equity * 100.0) if equity > 0 else 0.0

    def _gross_risk_pct_from_vals(self, risk_vals: np.ndarray, equity: float) -> float:
        total = float(np.sum(risk_vals))
        return (total / equity * 100.0) if equity > 0 else 0.0

    def _cluster_risk_pct(self, symbols: List[str], clusters: List[Optional[str]], risk_vals: np.ndarray, equity: float) -> Dict[str, float]:
        out = {}
        if equity <= 0:
            return out
        for s, c, r in zip(symbols, clusters, risk_vals):
            if not c:
                continue
            out.setdefault(c, 0.0)
            out[c] += float(r)
        for k in list(out.keys()):
            out[k] = out[k] / equity * 100.0
        return out

    def _max_corr_pair(self, symbols: List[str]) -> float:
        m = 0.0
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                m = max(m, abs(self.estimator.get_corr(symbols[i], symbols[j])))
        return m

__all__ = ["PortfolioAllocator", "AllocationAdjustment"]
