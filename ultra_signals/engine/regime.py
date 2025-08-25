from typing import Dict, Optional
from ultra_signals.core.custom_types import FeatureVector, RegimeFeatures
from ultra_signals.features.regime import classify_regime_full, RegimeStateMachine

_STATE = RegimeStateMachine()

class RegimeDetector:
    def __init__(self, settings: Dict):
        self.settings = settings

    def detect(self, fv: FeatureVector, *, spread_bps: Optional[float], volume_z: Optional[float]) -> RegimeFeatures:
        trend = fv.trend
        vol = fv.volatility
        volume_flow = fv.volume_flow
        adx = getattr(trend, "adx", None) if trend else None
        atr_pct = getattr(vol, "atr_percentile", None) if vol else None
        ema_sep_atr = None
        try:
            if trend and vol and getattr(vol, "atr", None):
                es = getattr(trend, "ema_short", None)
                el = getattr(trend, "ema_long", None)
                atr = getattr(vol, "atr", None)
                if all(v is not None for v in [es, el, atr]) and atr:
                    ema_sep_atr = abs(es - el) / atr
        except Exception:
            pass
        bb_width_pct_atr = None
        volume_z = getattr(volume_flow, "volume_z_score", None) if volume_flow else volume_z
        rf = classify_regime_full(
            adx,
            atr_pct,
            ema_sep_atr,
            self.settings,
            _STATE,
            bb_width_pct_atr=bb_width_pct_atr,
            volume_z=volume_z,
            spread_bps=spread_bps,
        )
        fv.regime = rf
        return rf

_detector: Optional[RegimeDetector] = None

def detect_regime(feature_vector: FeatureVector, settings: Dict) -> str:
    global _detector
    if _detector is None:
        _detector = RegimeDetector(settings)
    rf = _detector.detect(feature_vector, spread_bps=None, volume_z=None)
    return rf.profile.value