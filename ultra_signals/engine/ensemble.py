import logging
from typing import List, Dict, Tuple
from ultra_signals.core.custom_types import SubSignal, EnsembleDecision

logger = logging.getLogger(__name__)

# FIX: helper to safely derive both a signed score [-1,1] and a probability [0,1]
def _signed_and_prob_from_conf(s: SubSignal) -> Tuple[float, float]:
    """
    Returns (signed_confidence, probability) for a subsignal.
    - If confidence_calibrated is already 0..1, use that for prob; signed = map back to [-1,1] using direction.
    - If confidence_calibrated looks like a signed score outside 0..1, clamp to [-1,1] and map prob = (signed+1)/2.
    - If missing, fall back to s.score similarly; otherwise prob=0.5, signed=0.0.
    """
    # try calibrated first
    raw = getattr(s, "confidence_calibrated", None)
    if raw is None:
        raw = getattr(s, "score", 0.0)

    try:
        val = float(raw)
    except Exception:
        val = 0.0

    # Distinguish "prob-like" vs "signed-like"
    if 0.0 <= val <= 1.0:
        # treat as probability
        prob = val
        # derive a signed score consistent with direction
        signed = (prob * 2.0) - 1.0
    else:
        # treat as signed score; clamp to [-1,1] and map to prob
        signed = max(-1.0, min(1.0, val))
        prob = 0.5 * (signed + 1.0)

    # final clamps
    prob = max(0.0, min(1.0, prob))
    signed = max(-1.0, min(1.0, signed))
    return signed, prob


def _norm_dir(d: str) -> str:
    if not d:
        return "FLAT"
    d = str(d).upper()
    if d in ("LONG", "BUY"):
        return "LONG"
    if d in ("SHORT", "SELL"):
        return "SHORT"
    return "FLAT"


def combine_subsignals(
    subsignals: List[SubSignal], profile: str, settings: Dict
) -> EnsembleDecision:
    """
    Combines sub-signals into a single decision using weighted voting.

    Modes:
      - Legacy (your original math): set `ensemble.use_prob_mass: false`
      - Fixed prob-mass combiner (recommended): set `ensemble.use_prob_mass: true` (default)
    """
    if not subsignals:
        raise ValueError("Sub-signal list cannot be empty.")

    ts = subsignals[0].ts
    symbol = subsignals[0].symbol
    tf = subsignals[0].tf

    ensemble_settings = settings.get("ensemble", {})
    use_prob_mass = bool(ensemble_settings.get("use_prob_mass", True))  # default to fixed path

    # CHANGED: read vote_threshold from ensemble block first, then fallback to root, then default 0.5
    vote_threshold_raw = ensemble_settings.get(
        "vote_threshold",
        settings.get("vote_threshold", 0.5)
    )

    # Sprint 13: allow vote_threshold override via alpha_profiles weight_scale impact
    alpha_profiles = settings.get("alpha_profiles", {})
    prof_cfg = alpha_profiles.get(profile, {}) if isinstance(alpha_profiles, dict) else {}
    weight_scale = float(prof_cfg.get("weight_scale", 1.0))
    # If profile has higher weight_scale (>1), slightly lower threshold to be more aggressive
    if weight_scale > 1.0:
        try:
            base_thr_val = vote_threshold_raw if isinstance(vote_threshold_raw, float) else None
        except Exception:
            base_thr_val = None
        # Apply only when scalar (dict handled below)
        if base_thr_val is not None:
            vote_threshold_raw = max(0.0, base_thr_val * (1.0 - min(0.15, (weight_scale - 1.0) * 0.1)))

    # >>> SPRINT-8 ADDITIONS >>> -----------------------------------------------
    # Support regime-aware threshold (dict) or single float
    if isinstance(vote_threshold_raw, dict):
        vote_threshold = float(vote_threshold_raw.get(profile, vote_threshold_raw.get("mixed", 0.5)))
    else:
        vote_threshold = float(vote_threshold_raw)

    # Optional Sprint-8 params
    min_agree_by_reg = ensemble_settings.get("min_agree", {})  # may be dict per profile
    if isinstance(min_agree_by_reg, dict):
        min_agree = int(min_agree_by_reg.get(profile, min_agree_by_reg.get("mixed", 2)))
    else:
        # if configured as a single int/float
        try:
            min_agree = int(min_agree_by_reg)
        except Exception:
            min_agree = 2

    margin_of_victory = float(ensemble_settings.get("margin_of_victory", 0.10))
    confidence_floor   = float(ensemble_settings.get("confidence_floor", 0.60))
    # <<< SPRINT-8 ADDITIONS <<< -----------------------------------------------

    # NOTE: weights_profiles is your original key; keep it for backward-compat
    weights = settings.get("weights_profiles", {}).get(profile, {})
    # Sprint 11: allow flow metric specific weights to map into subsignals whose strategy_id matches
    flow_w_keys = ["cvd", "oi_rate", "liquidation_pulse", "depth_imbalance"]
    for k in flow_w_keys:
        if k not in weights:
            # default implicit weight kept at 1.0 later; we don't inject here
            continue
    logger.debug(f"Combining {len(subsignals)} subsignals with profile '{profile}' and weights: {weights}")
    for s in subsignals:
        logger.debug(f"  Subsignal: {s.strategy_id}, Direction: {s.direction}, Confidence: {s.confidence_calibrated}")

    # ----------------------------
    # Common accumulation section:
    # ----------------------------
    weighted_sum = 0.0       # legacy signed total (may be negative)
    long_voters = 0
    short_voters = 0
    total_weight = 0.0

    # Sprint-8 side-specific probability mass
    w_long = 0.0
    w_short = 0.0

    # Also keep normalized directions + prob for debug
    dbg_rows = []

    for s in subsignals:
        weight = weights.get(s.strategy_id, 1.0)
        if weight < 0:  # FIX: prevent negative weights from breaking sums
            weight = 0.0
        total_weight += weight

        # Normalize direction for safety
        s_dir = _norm_dir(getattr(s, "direction", "FLAT"))

        # FIX: use normalized pair (signed, prob)
        signed_c, prob_c = _signed_and_prob_from_conf(s)

        # LEGACY PATH accumulation (signed)
        direction_multiplier = 1 if s_dir == "LONG" else -1 if s_dir == "SHORT" else 0
        weighted_sum += weight * signed_c * direction_multiplier

        # voter counts (legacy)
        if s_dir == "LONG":
            long_voters += 1
        elif s_dir == "SHORT":
            short_voters += 1

        # Probability mass (non-negative)
        if s_dir == "LONG":
            w_long += weight * prob_c
        elif s_dir == "SHORT":
            w_short += weight * prob_c

        dbg_rows.append(f"{getattr(s,'strategy_id','?')}:{s_dir} p={prob_c:.3f} w={weight:.2f} signed={signed_c:.3f}")

    if total_weight > 0:
        normalized_sum = weighted_sum / total_weight
    else:
        normalized_sum = 0.0

    logger.debug(
        "Accumulated: total_weight=%.3f weighted_sum=%.3f normalized_sum=%.3f w_long=%.3f w_short=%.3f | subs=[%s]",
        total_weight, weighted_sum, normalized_sum, w_long, w_short, "; ".join(dbg_rows)
    )

    # ------------------------------------------------------------
    # MODE A) LEGACY decision using normalized_sum (kept intact)
    # ------------------------------------------------------------
    decision_dir_legacy = "FLAT"
    agree_count_legacy = 0
    if normalized_sum >= vote_threshold:
        decision_dir_legacy = "LONG"
        agree_count_legacy = long_voters
    elif normalized_sum <= -vote_threshold:
        decision_dir_legacy = "SHORT"
        agree_count_legacy = short_voters

    # ------------------------------------------------------------
    # MODE B) FIXED decision using probability mass (non-negative)
    # ------------------------------------------------------------
    # Winner mass & margin
    w_max = max(w_long, w_short)
    margin = abs(w_long - w_short)
    total_mass = w_long + w_short

    if w_long > w_short:
        side_pm = "LONG"
    elif w_short > w_long:
        side_pm = "SHORT"
    else:
        side_pm = "FLAT"

    # ensemble probability for the chosen side (clean 0..1)
    if total_mass > 0 and side_pm != "FLAT":
        p_ens = (w_long / total_mass) if side_pm == "LONG" else (w_short / total_mass)
    else:
        p_ens = 0.0

    # agree count for chosen side
    agree_pm = long_voters if side_pm == "LONG" else short_voters if side_pm == "SHORT" else 0

    # start from PM-based side and apply abstain gates
    decision_dir_pm = side_pm
    abstain_reason_pm = ""
    if decision_dir_pm != "FLAT":
        if margin < margin_of_victory:
            abstain_reason_pm = "LOW_MARGIN"
            decision_dir_pm = "FLAT"
            agree_pm = 0
        elif agree_pm < min_agree:
            abstain_reason_pm = "LOW_AGREE"
            decision_dir_pm = "FLAT"
            agree_pm = 0
        elif p_ens < confidence_floor:
            abstain_reason_pm = "LOW_CONF"
            decision_dir_pm = "FLAT"
            agree_pm = 0
    else:
        abstain_reason_pm = "TIE_OR_ZERO"

    # check vetoes (common to both modes)
    vetoes = [s.reasons.get("veto") for s in subsignals if getattr(s, "reasons", None) and s.reasons.get("veto")]
    if vetoes:
        logger.debug(f"Vetoes found: {vetoes}. Overriding decision to FLAT.")
        # Prefer to surface "VETO" only if we haven't abstained already
        if not abstain_reason_pm:
            abstain_reason_pm = "VETO"
        decision_dir_pm = "FLAT"
        agree_pm = 0

    # ------------------------------------------------------------
    # Select mode to output result
    # ------------------------------------------------------------
    if use_prob_mass:
        decision_dir = decision_dir_pm
        agree_count = agree_pm
        confidence = float(p_ens)  # clean 0..1
        # Keep your original normalized_sum for backward compatibility,
        # but also include PM telemetry so it's obvious in logs/UI.
        vote_detail = {
            "weighted_sum": round(normalized_sum, 3),     # legacy number (may be negative)
            "weighted_sum_pm": round(w_max, 3),           # non-negative winner mass
            "agree": agree_count,
            "total": len(subsignals),
            "profile": profile,
            "p_ens": round(p_ens, 3),
            "w_long": round(w_long, 3),
            "w_short": round(w_short, 3),
            "margin": round(margin, 3),
        }
        if abstain_reason_pm:
            vote_detail["reason"] = abstain_reason_pm
        # NEW: honor legacy vote_threshold using normalized_sum even in prob-mass mode AFTER building vote_detail
        if decision_dir == "LONG" and normalized_sum < vote_threshold:
            decision_dir = "FLAT"
            agree_count = 0
            confidence = 0.0
            vote_detail["reason"] = vote_detail.get("reason", "THR")
        elif decision_dir == "SHORT" and normalized_sum > -vote_threshold:
            decision_dir = "FLAT"
            agree_count = 0
            confidence = 0.0
            vote_detail["reason"] = vote_detail.get("reason", "THR")
        # Sprint 42: Macro risk-off gating (confidence dampen or veto)
        try:
            ca_cfg = (settings.get('cross_asset') or {}) if isinstance(settings, dict) else {}
            if ca_cfg.get('enabled'):
                # Macro features expected to be injected upstream (FeatureStore -> decision.vote_detail later)
                # Here we look for a hook in settings (optional lambda) or global context (not provided yet), so we only apply thresholds if present in settings cache
                macro_snapshot = settings.get('_latest_macro')  # optional injection point
                if isinstance(macro_snapshot, dict):
                    ro_prob = macro_snapshot.get('risk_off_prob')
                    regime = macro_snapshot.get('macro_risk_regime')
                    if ro_prob is not None and regime == 'risk_off' and decision_dir in ('LONG','SHORT'):
                        veto_thr = float(ca_cfg.get('risk_off_veto_prob', 0.72))
                        damp_thr = float(ca_cfg.get('risk_off_dampen_prob', 0.55))
                        damp_mult = float(ca_cfg.get('risk_off_conf_mult', 0.6))
                        if ro_prob >= veto_thr:
                            vote_detail.setdefault('macro', macro_snapshot)
                            vote_detail.setdefault('macro_action', 'VETO_RISK_OFF')
                            decision_dir = 'FLAT'
                            confidence = 0.0
                            vote_detail['reason'] = 'MACRO_RISK_OFF_VETO'
                        elif ro_prob >= damp_thr:
                            vote_detail.setdefault('macro', macro_snapshot)
                            vote_detail.setdefault('macro_action', 'DAMPEN_RISK_OFF')
                            confidence *= damp_mult
        except Exception:
            pass
        # Sprint 14: placeholder hook for orderflow weighting (final modulation happens in real_engine currently)
        # Could multiply confidence here in future if orderflow summary passed in subsignals.
        logger.debug(
            "[COMBINE:PM] side=%s p_ens=%.3f w_long=%.3f w_short=%.3f margin=%.3f thr=%.2f conf_floor=%.2f reason=%s",
            side_pm, p_ens, w_long, w_short, margin, vote_threshold, confidence_floor, vote_detail.get("reason", "")
        )
    else:
        # Legacy output exactly like before (with your Sprint-8 abstains still applied above)
        decision_dir = decision_dir_legacy
        agree_count = agree_count_legacy
        # your previous confidence: magnitude of winner’s weight, clipped [0,1]
        confidence = min(1.0, abs(w_max))
        vote_detail = {
            "weighted_sum": round(normalized_sum, 3),
            "agree": agree_count,
            "total": len(subsignals),
            "profile": profile,
        }
        # try to reflect PM abstain reason if legacy decided a side but PM would abstain
        # (optional, but helpful for debugging)
        if decision_dir == "FLAT" and abstain_reason_pm:
            vote_detail["reason"] = abstain_reason_pm

    logger.info(f"Final decision for {symbol} at {ts}: {decision_dir}, Confidence: {confidence:.2f}, Vote Detail: {vote_detail}")
    logger.debug(f"Final decision: {decision_dir}, Confidence: {confidence:.2f}, Vote Detail: {vote_detail}")

    logger.debug(
        "Ensemble: profile=%s votes=%s total_weight=%.3f long_w=%.3f short_w=%.3f "
        "normalized=%.3f min_score=%.3f majority=%.2f decision=%s",
        profile,
        vote_detail,
        total_weight,
        sum(weights.get(s.strategy_id, 1.0) for s in subsignals if _norm_dir(getattr(s, 'direction', 'FLAT')) == "LONG"),
        sum(weights.get(s.strategy_id, 1.0) for s in subsignals if _norm_dir(getattr(s, 'direction', 'FLAT')) == "SHORT"),
        normalized_sum,
        ensemble_settings.get("min_score", 0.1),
        ensemble_settings.get("majority_threshold", 0.51),
        decision_dir,
    )
    # Attach threshold used for transparency (Sprint 13 visualization requirement)
    try:
        vote_detail.setdefault("vote_threshold", round(vote_threshold, 4))
    except Exception:
        pass

    return EnsembleDecision(
        ts=ts,
        symbol=symbol,
        tf=tf,
        decision=decision_dir,
        confidence=confidence,
        subsignals=subsignals,
        vote_detail=vote_detail,
        vetoes=vetoes
    )
