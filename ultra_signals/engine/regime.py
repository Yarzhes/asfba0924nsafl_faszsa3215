from typing import Dict
from ultra_signals.core.custom_types import FeatureVector

def detect_regime(feature_vector: FeatureVector, settings: Dict) -> str:
    """
    Determines the market regime based on feature values.
    
    This is a simplified example. A real implementation would be more complex.
    """
    # Default to "chop" if not enough data
    if not all([
        feature_vector.trend,
        feature_vector.trend.adx,
        feature_vector.volatility,
        feature_vector.volatility.atr_percentile
    ]):
        return "chop"

    adx = feature_vector.trend.adx
    atr_percentile = feature_vector.volatility.atr_percentile

    # Simple rule-based regime detection
    if adx > settings.get("adx_min_trend", 25) and atr_percentile > 0.6:
        return "trend"
    elif atr_percentile < 0.4:
        return "mean_revert"
    else:
        return "chop"