import logging
from typing import List, Dict
from ultra_signals.core.custom_types import SubSignal, EnsembleDecision

logger = logging.getLogger(__name__)

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
    vote_threshold = float(settings.get("vote_threshold", 0.5))
    
    weights = settings.get("weights_profiles", {}).get(profile, {})
    logger.debug(f"Combining {len(subsignals)} subsignals with profile '{profile}' and weights: {weights}")
    for s in subsignals:
        logger.debug(f"  Subsignal: {s.strategy_id}, Direction: {s.direction}, Confidence: {s.confidence_calibrated}")

    weighted_sum = 0.0
    long_voters = 0
    short_voters = 0
    total_weight = 0

    for s in subsignals:
        weight = weights.get(s.strategy_id, 1.0)
        total_weight += weight
        c = s.confidence_calibrated
        direction_multiplier = 1 if s.direction == "LONG" else -1 if s.direction == "SHORT" else 0
        weighted_sum += weight * c * direction_multiplier
        if s.direction == "LONG":
            long_voters += 1
        elif s.direction == "SHORT":
            short_voters += 1

    if total_weight > 0:
        normalized_sum = weighted_sum / total_weight
    else:
        normalized_sum = 0
    
    logger.debug(f"Total weight: {total_weight:.2f}, Weighted sum: {weighted_sum:.2f}, Normalized sum: {normalized_sum:.2f}")

    if normalized_sum >= vote_threshold:
        decision_dir = "LONG"
        agree_count = long_voters
    elif normalized_sum <= -vote_threshold:
        decision_dir = "SHORT"
        agree_count = short_voters
    else:
        decision_dir = "FLAT"
        agree_count = 0

    vetoes = [s.reasons.get("veto") for s in subsignals if s.reasons and s.reasons.get("veto")]
    if vetoes:
        logger.debug(f"Vetoes found: {vetoes}. Overriding decision to FLAT.")
        decision_dir = "FLAT"

    confidence = min(1.0, abs(normalized_sum))
    
    vote_detail = {
        "weighted_sum": round(normalized_sum, 3),
        "agree": agree_count,
        "total": len(subsignals),
        "profile": profile,
    }
    
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