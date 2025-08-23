import logging
from typing import Dict, Optional

import pandas as pd

from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import EnsembleDecision, FeatureVector, SubSignal
from ultra_signals.engine import ensemble, regime, scoring

logger = logging.getLogger(__name__)


class RealSignalEngine:
    """A realistic signal engine that uses the scoring and ensemble logic."""

    def __init__(self, settings: Dict, feature_store: FeatureStore):
        self.settings = settings
        self.feature_store = feature_store

    def generate_signal(
        self, ohlcv_segment: pd.DataFrame, symbol: str
    ) -> Optional[EnsembleDecision]:
        """Generates a signal for the given bar by building a full feature vector."""

        latest_bar = ohlcv_segment.iloc[-1]
        timestamp = latest_bar.name

        # 1. Get all features from the store for the current timestamp
        features = self.feature_store.get_features(symbol, timestamp)
        if not features:
            logger.debug(f"Not enough data to compute features for {symbol} at {timestamp}")
            return None

        # 2. Construct the full feature vector
        feature_vector = FeatureVector(
            symbol=symbol,
            timeframe=self.settings["runtime"]["primary_timeframe"],
            ohlcv=latest_bar.to_dict(),
            # The following now come from the feature store
            trend=features.get("trend"),
            momentum=features.get("momentum"),
            volatility=features.get("volatility"),
            volume_flow=features.get("volume_flow"),
            # Mock derivatives/orderbook for now as they are not computed
            derivatives=None,
            orderbook=None,
            rs=None,
        )
        logger.debug(f"FV for {symbol} at {timestamp}: {feature_vector.model_dump_json(indent=2)}")

        # 3. Score the components based on the feature vector
        component_scores = scoring.component_scores(
            feature_vector, self.settings["features"]
        )
        logger.debug(f"Component scores: {component_scores}")
        
        # 4. Create sub-signals from scores
        subsignals = []
        for name, score in component_scores.items():
            if pd.isna(score) or score == 0:
                continue

            direction = "FLAT"
            if score > 0:
                direction = "LONG"
            elif score < 0:
                direction = "SHORT"

            subsignals.append(
                SubSignal(
                    ts=int(timestamp.timestamp()),
                    symbol=symbol,
                    tf=feature_vector.timeframe,
                    strategy_id=name,
                    direction=direction,
                    confidence_raw=abs(score),
                    confidence_calibrated=abs(score),  # No calibration
                    reasons={},
                )
            )

        if not subsignals:
            logger.debug("No subsignals generated, returning None.")
            return None

        # 5. Detect the regime dynamically
        current_regime = regime.detect_regime(feature_vector, self.settings["regime"])
        logger.debug(f"Detected regime: {current_regime}")

        # 6. Combine subsignals into a final decision
        try:
            logger.debug(
                "Subsignals: %s",
                [(s.strategy_id, getattr(s, "confidence_calibrated", None), getattr(s, "direction", None)) for s in subsignals]
            )
        except Exception:
            logger.debug("Subsignals present but not printable")
        final_decision = ensemble.combine_subsignals(
            subsignals, current_regime, self.settings
        )

        if final_decision:
            if final_decision.decision != "FLAT":
                logger.info(f"Subsignals for {symbol} at {timestamp}: {subsignals}")
                logger.info(f"Final Decision: {final_decision.decision} @ {final_decision.confidence} (Vote Detail: {final_decision.vote_detail})")
            else:
                logger.debug(f"Final Decision for {symbol} at {timestamp}: FLAT (Vote Detail: {final_decision.vote_detail})")
        
        return final_decision

    def should_exit(self, symbol, pos, bar, features):
        px = bar["close"]
        atr = getattr(features.get("volatility"), "atr", None) or 0
        # Default multipliers if not already on the position/config
        k_stop = getattr(pos, "atr_mult_stop", 2.0)
        k_tp   = getattr(pos, "atr_mult_tp",   3.0)
        max_bars = getattr(pos, "max_bars", 288)  # e.g., 24h on 5m bars

        if pos.side == "LONG":
            if atr and px <= pos.entry_price - k_stop*atr: return "STOP"
            if atr and px >= pos.entry_price + k_tp*atr:   return "TP"
        else:  # SHORT
            if atr and px >= pos.entry_price + k_stop*atr: return "STOP"
            if atr and px <= pos.entry_price - k_tp*atr:   return "TP"

        # Time stop
        if pos.bars_held >= max_bars:
            return "TIME_STOP"
        return None