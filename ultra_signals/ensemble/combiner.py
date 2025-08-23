from dataclasses import replace
from typing import List, Dict
from ultra_signals.core.custom_types import SubSignal, EnsembleDecision

DIRECTION_MAP = {"LONG": 1.0, "SHORT": -1.0, "FLAT": 0.0}

def _get_weight(strategy_id: str, profile: str, settings: dict) -> float:
    ensemble_settings = settings.get("ensemble", {})
    weights = ensemble_settings.get("weights", {})
    profile_weights = weights.get(profile, weights.get("default", {}))
    return float(profile_weights.get(strategy_id, 1.0))

def combine_subsignals(subsignals: List[SubSignal], profile: str, settings: dict) -> EnsembleDecision:
    """
    Combines sub-signals into a single decision using weighted voting.
    """
    ensemble_settings = settings.get("ensemble", {})
    vote_threshold = float(ensemble_settings.get("vote_threshold", 0.5))
    
    weighted_sum = 0.0
    long_voters = 0
    short_voters = 0

    for s in subsignals:
        w = _get_weight(s.strategy_id, profile, settings)
        c = s.confidence_calibrated
        direction_multiplier = DIRECTION_MAP.get(s.direction, 0.0)
        weighted_sum += w * c * direction_multiplier
        if s.direction == "LONG":
            long_voters += 1
        elif s.direction == "SHORT":
            short_voters += 1

    if weighted_sum >= vote_threshold:
        decision_dir = "LONG"
        agree_count = long_voters
    elif weighted_sum <= -vote_threshold:
        decision_dir = "SHORT"
        agree_count = short_voters
    else:
        decision_dir = "FLAT"
        agree_count = 0

    vetoes = [s.reasons.get("veto") for s in subsignals if s.reasons.get("veto")]
    if vetoes:
        decision_dir = "FLAT"

    confidence = min(1.0, abs(weighted_sum))
    
    vote_detail = {
        "weighted_sum": round(weighted_sum, 3),
        "agree": agree_count,
        "total": long_voters + short_voters,
        "profile": profile,
    }

    return EnsembleDecision(
        ts=subsignals[0].ts,
        symbol=subsignals[0].symbol,
        tf=subsignals[0].tf,
        decision=decision_dir,
        confidence=confidence,
        subsignals=subsignals,
        vote_detail=vote_detail,
        vetoes=vetoes
    )