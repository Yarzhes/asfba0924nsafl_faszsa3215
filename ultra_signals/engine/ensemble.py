import logging
from typing import List, Dict
from ultra_signals.core.custom_types import SubSignal, EnsembleDecision

logger = logging.getLogger(__name__)

# FIX: helper to safely derive both a signed score [-1,1] and a probability [0,1]
def _signed_and_prob_from_conf(s: SubSignal) -> (float, float):
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


def combine_subsignals(
    subsignals: List[SubSignal], profile: str, settings: Dict
) -> EnsembleDecision:
    """
    Combines sub-signals into a single decision using weighted voting.
    """
    if not subsignals:
        raise ValueError("Sub-signal list cannot be empty.")

    ts = subsignals[0].ts
    symbol = subsignals[0].symbol
    tf = subsignals[0].tf

    ensemble_settings = settings.get("ensemble", {})
    # CHANGED: read vote_threshold from ensemble block first, then fallback to root, then default 0.5
    vote_threshold_raw = ensemble_settings.get(
        "vote_threshold", 
        settings.get("vote_threshold", 0.5)
    )

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

    weights = settings.get("weights_profiles", {}).get(profile, {})
    logger.debug(f"Combining {len(subsignals)} subsignals with profile '{profile}' and weights: {weights}")
    for s in subsignals:
        logger.debug(f"  Subsignal: {s.strategy_id}, Direction: {s.direction}, Confidence: {s.confidence_calibrated}")

    weighted_sum = 0.0
    long_voters = 0
    short_voters = 0
    total_weight = 0

    # >>> SPRINT-8 ADDITIONS >>> -----------------------------------------------
    # Track side-specific weighted sums for margin-of-victory logic
    w_long = 0.0
    w_short = 0.0
    # <<< SPRINT-8 ADDITIONS <<< -----------------------------------------------

    for s in subsignals:
        weight = weights.get(s.strategy_id, 1.0)
        # FIX: prevent negative weights from breaking sums
        if weight < 0:
            weight = 0.0
        total_weight += weight

        # FIX: use normalized pair (signed, prob)
        signed_c, prob_c = _signed_and_prob_from_conf(s)

        direction_multiplier = 1 if s.direction == "LONG" else -1 if s.direction == "SHORT" else 0
        # keep original direction math for normalized_sum (uses signed confidence)
        weighted_sum += weight * signed_c * direction_multiplier

        # Count voters for original logic
        if s.direction == "LONG":
            long_voters += 1
        elif s.direction == "SHORT":
            short_voters += 1

        # >>> SPRINT-8 ADDITIONS >>> -------------------------------------------
        # Side-specific weighted totals (use probability so they are non-negative)
        if s.direction == "LONG":
            w_long += weight * prob_c
        elif s.direction == "SHORT":
            w_short += weight * prob_c
        # <<< SPRINT-8 ADDITIONS <<< -------------------------------------------

    if total_weight > 0:
        normalized_sum = weighted_sum / total_weight
    else:
        normalized_sum = 0.0
    
    logger.debug(f"Total weight: {total_weight:.2f}, Weighted sum: {weighted_sum:.2f}, Normalized sum: {normalized_sum:.2f}")

    # Original thresholding using normalized_sum
    if normalized_sum >= vote_threshold:
        decision_dir = "LONG"
        agree_count = long_voters
    elif normalized_sum <= -vote_threshold:
        decision_dir = "SHORT"
        agree_count = short_voters
    else:
        decision_dir = "FLAT"
        agree_count = 0

    # >>> SPRINT-8 ADDITIONS >>> -----------------------------------------------
    # Sprint-8 abstain rules layered on top:
    # - Compute the winning side's raw weight (w_max) and margin
    w_max = max(w_long, w_short)
    margin = abs(w_long - w_short)

    # If we picked LONG/SHORT above, enforce extra abstain conditions
    abstain_reason = ""

    if decision_dir != "FLAT":
        # Require clear margin of victory
        if margin < margin_of_victory:
            abstain_reason = "LOW_MARGIN"
            decision_dir = "FLAT"
            agree_count = 0

        # Require minimum agreement count (per profile)
        elif agree_count < min_agree:
            abstain_reason = "LOW_AGREE"
            decision_dir = "FLAT"
            agree_count = 0

        # Require minimum ensemble probability (confidence floor)
        # We use w_max (sum of weight*prob) as proxy for ensemble prob → non-negative
        elif w_max < confidence_floor:
            abstain_reason = "LOW_CONF"
            decision_dir = "FLAT"
            agree_count = 0
    # <<< SPRINT-8 ADDITIONS <<< -----------------------------------------------

    vetoes = [s.reasons.get("veto") for s in subsignals if getattr(s, "reasons", None) and s.reasons.get("veto")]
    if vetoes:
        logger.debug(f"Vetoes found: {vetoes}. Overriding decision to FLAT.")
        decision_dir = "FLAT"
        # >>> SPRINT-8 ADDITIONS >>> -------------------------------------------
        # If we abstained due to veto, prefer to surface reason as VETO
        abstain_reason = abstain_reason or "VETO"
        # <<< SPRINT-8 ADDITIONS <<< -------------------------------------------

    # >>>>>> CHANGE APPLIED (Fix #3): confidence from |w_max|, not normalized_sum >>>>>>
    # Confidence should reflect the magnitude of the winning side’s weight, clipped to [0, 1].
    confidence = min(1.0, abs(w_max))
    # <<<<<< CHANGE APPLIED <<<<<<

    vote_detail = {
        # NOTE: keep normalized_sum here for telemetry/UI; tests look for this number.
        "weighted_sum": round(normalized_sum, 3),
        "agree": agree_count,
        "total": len(subsignals),
        "profile": profile,
    }

    # >>> SPRINT-8 ADDITIONS >>> -----------------------------------------------
    # Also include "reason" so downstream formatters can show why we abstained
    if abstain_reason:
        vote_detail["reason"] = abstain_reason
    # <<< SPRINT-8 ADDITIONS <<< -----------------------------------------------

    logger.info(f"Final decision for {symbol} at {ts}: {decision_dir}, Confidence: {confidence:.2f}, Vote Detail: {vote_detail}")
    logger.debug(f"Final decision: {decision_dir}, Confidence: {confidence:.2f}, Vote Detail: {vote_detail}")

    logger.debug(
        "Ensemble: profile=%s votes=%s total_weight=%.3f long_w=%.3f short_w=%.3f "
        "normalized=%.3f min_score=%.3f majority=%.2f decision=%s",
        profile,
        vote_detail,
        total_weight,
        sum(weights.get(s.strategy_id, 1.0) for s in subsignals if s.direction == "LONG"),
        sum(weights.get(s.strategy_id, 1.0) for s in subsignals if s.direction == "SHORT"),
        normalized_sum,
        ensemble_settings.get("min_score", 0.1),
        ensemble_settings.get("majority_threshold", 0.51),
        decision_dir,
    )
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
