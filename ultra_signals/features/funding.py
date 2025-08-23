# -*- coding: utf-8 -*-
"""
Feature computation based on historical funding rate data.
"""
from typing import Dict, TYPE_CHECKING
from ultra_signals.core.custom_types import Symbol

if TYPE_CHECKING:
    from ultra_signals.core.feature_store import FeatureStore


def compute_funding_features(feature_store: "FeatureStore", symbol: Symbol) -> Dict:
    """
    Computes features based on funding rate history.

    :param feature_store: The feature store instance.
    :param symbol: The symbol to compute features for.
    :return: A dictionary of computed funding features.
    """
    funding_history = feature_store.get_funding_rate_history(symbol)
    if not funding_history:
        return {"funding_rate_avg_8p": 0.0}

    total_rate = sum(item.get("funding_rate", 0.0) for item in funding_history)
    avg_rate = total_rate / len(funding_history) if funding_history else 0.0

    return {"funding_rate_avg_8p": avg_rate}