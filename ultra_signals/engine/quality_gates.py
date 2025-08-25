"""Sprint 18: Quality Gates (Confidence Binning + Targeted Vetoes)
-----------------------------------------------------------------
Provides a single entrypoint `QualityGates.evaluate(...)` that returns a
QualityDecision object with:
  - qscore (0..1 composite quality score)
  - bin (A+/A/B/C/D)
  - hard veto reasons (blocked flag)
  - soft flags (non-blocking but may impose extra confirmation requirements)
  - size multiplier from bin action policy

Design Principles
-----------------
* Fail safe: any missing critical input -> conservative outcome (lower bin or veto)
* All thresholds & weights configurable under settings['quality_gates']
* Minimal coupling: inputs are plain dict / simple objects already produced in engine

Expected settings snippet (see settings.yaml addition):
quality_gates:
  enabled: true
  qscore_bins: { Aplus: 0.85, A: 0.75, B: 0.65, C: 0.55 }
  bin_actions:
    Aplus: { size_mult: 1.30, require_extra_confirm: false }
    A:     { size_mult: 1.15, require_extra_confirm: false }
    B:     { size_mult: 1.00, require_extra_confirm: false }
    C:     { size_mult: 0.75, require_extra_confirm: true }
    D:     { size_mult: 0.00, require_extra_confirm: true }
  veto: { ... }
  soft: { ... }

Inputs
------
features: dict of feature objects (trend, momentum, volatility, regime, flow_metrics, etc.)
decision: EnsembleDecision (already includes confidence post-orderflow boost)
playbook: dict or object with expected_rr and other plan metadata (optional)
settings: global settings dict

Simplifications
---------------
* Orderflow alignment score attempts to read cvd_chg / liq impulses from vote_detail['orderflow'] if present.
* Regime-fit score: 1.0 if playbook selected and its regime matches current regime profile, else 0.6 fallback, else 0.4 if mismatch.
* Expected RR scaling: min-max scale between configured rr_min (from playbook risk if available) and (rr_min + 2.0) else clamp.
* Many microstructure details (heatmap proximity, slippage model) rely on fields in features or vote_detail; if absent we skip or conservative flag.

Author: Sprint 18 implementation.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from loguru import logger
from .orderflow import OrderFlowSnapshot  # type: ignore
from ultra_signals.core.custom_types import EnsembleDecision, QualityDecision

# Lightweight helpers ---------------------------------------------------------

def _safe_float(x, default=None):
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default


def _min_max_scale(val: float, vmin: float, vmax: float) -> float:
    if val is None or vmin is None or vmax is None or vmax <= vmin:
        return 0.0
    return max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))


@dataclass
class _GateContext:
    features: Dict[str, Any]
    decision: EnsembleDecision
    playbook: Any
    settings: Dict[str, Any]
    qcfg: Dict[str, Any]


class QualityGates:
    @staticmethod
    def evaluate(features: Dict[str, Any], decision: EnsembleDecision, playbook: Any, settings: Dict[str, Any]) -> QualityDecision:
        qcfg = (settings.get('quality_gates') or {}) if isinstance(settings, dict) else {}
        if not qcfg.get('enabled', True):
            # pass-through (treat as neutral B bin w/o changes)
            return QualityDecision(bin='B', qscore=float(decision.confidence or 0.0), blocked=False, size_multiplier=1.0)
        ctx = _GateContext(features, decision, playbook, settings, qcfg)

        qscore = QualityGates._compute_qscore(ctx)
        bin_label = QualityGates._assign_bin(qscore, qcfg)
        blocked, hard_reasons, soft_flags = QualityGates._apply_vetoes(ctx, qscore, bin_label)
        action = (qcfg.get('bin_actions') or {}).get(_canonical_bin_key(bin_label), {})
        size_mult = float(action.get('size_mult', 1.0))
        require_extra = bool(action.get('require_extra_confirm', False))

        requirements: Dict[str, bool] = {}
        notes_parts: List[str] = []
        if require_extra and not blocked:
            # Soft flag types that translate into requirements
            if 'LATE_MOVE' in soft_flags:
                requirements['need_additional_confirm'] = True
                notes_parts.append('late_move_requires_confirm')
            if 'OFI_CONFLICT' in soft_flags:
                requirements['need_ofi_resolution'] = True
            if 'LOW_LIQ' in soft_flags:
                requirements['need_liq_confirm'] = True
            if 'RR_WEAK' in soft_flags:
                requirements['need_rr_confirm'] = True
        if blocked:
            notes_parts.append('blocked')

        notes = ';'.join(notes_parts)
        return QualityDecision(
            bin=bin_label,
            qscore=round(qscore, 4),
            blocked=blocked or bin_label == 'D',
            veto_reasons=hard_reasons,
            soft_flags=soft_flags,
            requirements=requirements,
            size_multiplier=size_mult if not blocked else 0.0,
            notes=notes
        )

    # -- Composite Score -----------------------------------------------------
    @staticmethod
    def _compute_qscore(ctx: _GateContext) -> float:
        dec = ctx.decision
        features = ctx.features
        playbook = ctx.playbook or {}
        qcfg = ctx.qcfg

        # Weights (defaults per spec)
        w_conf = float(qcfg.get('weights', {}).get('ensemble_conf', 0.40))
        w_of = float(qcfg.get('weights', {}).get('orderflow', 0.20))
        w_regime = float(qcfg.get('weights', {}).get('regime_fit', 0.20))
        w_rr = float(qcfg.get('weights', {}).get('expected_rr', 0.20))

        confidence = float(dec.confidence or 0.0)
        orderflow_score = QualityGates._orderflow_alignment(dec)
        regime_fit = QualityGates._regime_fit_score(features, playbook)
        rr_score = QualityGates._rr_score(playbook, qcfg)

        qscore = (w_conf * confidence) + (w_of * orderflow_score) + (w_regime * regime_fit) + (w_rr * rr_score)
        return max(0.0, min(1.0, qscore))

    @staticmethod
    def _orderflow_alignment(decision: EnsembleDecision) -> float:
        of = (decision.vote_detail or {}).get('orderflow') if isinstance(decision.vote_detail, dict) else None
        if not of:
            return 0.5  # neutral
        direction = decision.decision
        cvd_chg = _safe_float(of.get('cvd_chg'))
        liq_dom = of.get('liq_dom')
        liq_impulse = _safe_float(of.get('liq_impulse'))
        sweep_side = of.get('sweep_side')
        score = 0.5
        if direction == 'LONG':
            if cvd_chg and cvd_chg > 0: score += 0.15
            if liq_dom == 'short': score += min(0.15, (liq_impulse or 0)/2.0)
            if sweep_side == 'bid': score += 0.10
            if sweep_side == 'ask': score -= 0.10
        elif direction == 'SHORT':
            if cvd_chg and cvd_chg < 0: score += 0.15
            if liq_dom == 'long': score += min(0.15, (liq_impulse or 0)/2.0)
            if sweep_side == 'ask': score += 0.10
            if sweep_side == 'bid': score -= 0.10
        return max(0.0, min(1.0, score))

    @staticmethod
    def _regime_fit_score(features: Dict[str, Any], playbook: Any) -> float:
        regime_obj = features.get('regime') if isinstance(features, dict) else None
        if not regime_obj:
            return 0.5
        try:
            regime_profile = getattr(regime_obj, 'profile', None) or getattr(regime_obj, 'mode', None)
        except Exception:
            regime_profile = None
        if not playbook:
            return 0.6  # some regime but no playbook context
        pb_name = getattr(playbook, 'name', None) if not isinstance(playbook, dict) else playbook.get('name')
        # crude mapping: if playbook name contains regime keyword treat as fit
        if pb_name and regime_profile and str(regime_profile).lower() in str(pb_name).lower():
            return 1.0
        return 0.6 if regime_profile else 0.4

    @staticmethod
    def _rr_score(playbook: Any, qcfg: Dict[str, Any]) -> float:
        if not playbook:
            return 0.5
        # expected rr extracted from plan / playbook risk config
        exp_rr = None
        if isinstance(playbook, dict):
            exp_rr = playbook.get('expected_rr') or playbook.get('rr')
            rr_min = (playbook.get('risk') or {}).get('rr_min') if isinstance(playbook.get('risk'), dict) else playbook.get('rr_min')
        else:
            exp_rr = getattr(playbook, 'expected_rr', None)
            rr_min = getattr(getattr(playbook, 'risk', None), 'rr_min', None)
        exp_rr = _safe_float(exp_rr, 0.0)
        rr_min = _safe_float(rr_min, 1.0) or 1.0
        rr_max = rr_min + 2.0  # simple dynamic window
        return _min_max_scale(exp_rr, rr_min, rr_max)

    # -- Binning -------------------------------------------------------------
    @staticmethod
    def _assign_bin(qscore: float, qcfg: Dict[str, Any]) -> str:
        bins = qcfg.get('qscore_bins', {})
        # ensure numeric
        aplus = float(bins.get('Aplus', 0.85))
        a_ = float(bins.get('A', 0.75))
        b_ = float(bins.get('B', 0.65))
        c_ = float(bins.get('C', 0.55))
        if qscore >= aplus: return 'A+'
        if qscore >= a_: return 'A'
        if qscore >= b_: return 'B'
        if qscore >= c_: return 'C'
        return 'D'

    # -- Veto evaluation -----------------------------------------------------
    @staticmethod
    def _apply_vetoes(ctx: _GateContext, qscore: float, bin_label: str) -> Tuple[bool, List[str], List[str]]:
        hard: List[str] = []
        soft: List[str] = []
        veto_cfg = ctx.qcfg.get('veto', {})
        soft_cfg = ctx.qcfg.get('soft', {})
        feats = ctx.features
        dec = ctx.decision

        # Spread wide hard veto
        spread_pct = None
        try:
            # attempt to read from flow metrics or vote_detail
            fm = feats.get('flow_metrics')
            if fm is not None:
                spread_pct = getattr(fm, 'spread_bps', None)
                if spread_pct is not None:
                    spread_pct = spread_pct / 10000.0  # bps to frac
        except Exception:
            pass
        max_spread_pct = _safe_float(veto_cfg.get('max_spread_pct'), 0.06)
        if spread_pct is not None and spread_pct > max_spread_pct:
            hard.append('SPREAD_WIDE')

        # ATR percentile volatility spike
        atr_pct = None
        try:
            reg = feats.get('regime')
            if reg is not None:
                atr_pct = getattr(reg, 'atr_percentile', None)
            if atr_pct is None:
                vol = feats.get('volatility')
                atr_pct = getattr(vol, 'atr_percentile', None) if vol is not None else None
        except Exception:
            pass
        atr_limit = _safe_float(veto_cfg.get('atr_pct_limit'), 0.97)
        if atr_pct is not None and atr_pct >= atr_limit:
            hard.append('VOL_SPIKE')

        # Book staleness (if feature store exposes timestamps)
        book_stale_ms = feats.get('orderbook', {}).get('age_ms') if isinstance(feats.get('orderbook'), dict) else None
        max_stale = _safe_float(veto_cfg.get('max_book_staleness_ms'), 2500)
        if book_stale_ms is not None and book_stale_ms > max_stale:
            hard.append('BOOK_STALE')

        # Data gap: expect maybe features contain 'missing_bars'
        missing_bars = feats.get('meta', {}).get('missing_bars') if isinstance(feats.get('meta'), dict) else None
        max_missing = int(veto_cfg.get('max_missing_bars', 2))
        if missing_bars is not None and missing_bars > max_missing:
            hard.append('DATA_GAP')

        # Correlation stack (placeholder: rely on decision.vote_detail['correlation'])
        corr_detail = (dec.vote_detail or {}).get('correlation') if isinstance(dec.vote_detail, dict) else None
        if corr_detail and corr_detail.get('stack_block'):
            hard.append('CORR_STACK')

        # Daily drawdown / streak (pull from settings.brakes maybe - placeholder flags in feats)
        if feats.get('risk', {}).get('daily_dd_violation') if isinstance(feats.get('risk'), dict) else False:
            hard.append('DAILY_DD')

        # Heatmap proximity (e.g., distance bp to large liquidation wall) from position_sizer or heatmap feature
        liqmap = (dec.vote_detail or {}).get('position_sizer', {}) if isinstance(dec.vote_detail, dict) else {}
        heatmap_bp = _safe_float(liqmap.get('nearest_wall_bp'))
        near_bp = _safe_float(veto_cfg.get('near_heatmap_bp'), 8)
        if heatmap_bp is not None and heatmap_bp <= near_bp:
            hard.append('HEATMAP_NEAR')

        # Slippage risk: expected_rr drop beyond limit (need playbook expected_rr + slippage_adjusted_rr maybe)
        if ctx.playbook and isinstance(ctx.playbook, dict):
            exp_rr = _safe_float(ctx.playbook.get('expected_rr'))
            slip_rr = _safe_float(ctx.playbook.get('slippage_rr'))
            if exp_rr is not None and slip_rr is not None:
                drop = exp_rr - slip_rr
                max_drop = _safe_float(veto_cfg.get('max_slippage_rr_drop'), 0.2)
                if drop is not None and drop > max_drop:
                    hard.append('SLIPPAGE_RISK')

        # ---------------- Soft Gates ----------------
        fm = feats.get('flow_metrics')
        if fm is not None:
            ofi = getattr(fm, 'ofi', None)
            ofi_limit = _safe_float(soft_cfg.get('ofi_conflict_limit'), -0.15)
            if ofi is not None and dec.decision == 'LONG' and ofi < ofi_limit:
                soft.append('OFI_CONFLICT')
            if ofi is not None and dec.decision == 'SHORT' and ofi > -ofi_limit:
                soft.append('OFI_CONFLICT')
            vol_z = getattr(fm, 'volume_z', None)
            min_vol_z = _safe_float(soft_cfg.get('min_volume_z'), -0.8)
            if vol_z is not None and vol_z < min_vol_z:
                soft.append('LOW_LIQ')
        # Late move: compare current price distance vs ATR (need ohlcv close + maybe playbook anchor entry)
        try:
            ohlcv = feats.get('ohlcv') or dec.vote_detail.get('ohlcv') if isinstance(dec.vote_detail, dict) else None
            close_px = _safe_float((ohlcv or {}).get('close'))
            trigger_px = _safe_float((ctx.playbook or {}).get('trigger_price'))
            atr_val = _safe_float(getattr(feats.get('volatility'), 'atr', None)) if feats.get('volatility') else None
            late_atr = _safe_float(soft_cfg.get('late_move_atr'), 0.8)
            if close_px is not None and trigger_px is not None and atr_val and atr_val > 0:
                if abs(close_px - trigger_px) >= late_atr * atr_val:
                    soft.append('LATE_MOVE')
        except Exception:
            pass

        # RR Weak: expected rr between absolute min and preferred
        if ctx.playbook and isinstance(ctx.playbook, dict):
            exp_rr = _safe_float(ctx.playbook.get('expected_rr'))
            rr_min = _safe_float((ctx.playbook.get('risk') or {}).get('rr_min'))
            if exp_rr and rr_min and exp_rr < (rr_min + 0.15):
                soft.append('RR_WEAK')

        blocked = len(hard) > 0
        return blocked, hard, soft

# ---------------------------------------------------------------------------

def _canonical_bin_key(bin_label: str) -> str:
    return 'Aplus' if bin_label == 'A+' else bin_label

__all__ = ['QualityGates']
