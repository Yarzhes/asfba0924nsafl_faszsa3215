# ultra_signals/strategy/minimal_engine.py
from typing import Optional
import pandas as pd
from ultra_signals.core.custom_types import EnsembleDecision

class MinimalSignalEngine:
    """
    Super-simple example: 
    - LONG if close > open
    - SHORT if close < open
    - FLAT if equal
    This is just to verify the event loop opens trades.
    """

    def generate_signal(self, ohlcv_segment: pd.DataFrame, symbol: str) -> Optional[EnsembleDecision]:
        ts = int(ohlcv_segment.index[-1].timestamp())
        last = ohlcv_segment.iloc[-1]
        close = float(last["close"])
        open_ = float(last["open"])

        if close > open_:
            decision = "LONG"
            conf = 0.7
        elif close < open_:
            decision = "SHORT"
            conf = 0.7
        else:
            decision = "FLAT"
            conf = 0.0

        return EnsembleDecision(
            ts=ts,
            symbol=symbol,
            tf="auto",
            decision=decision,
            confidence=conf,
            subsignals=[],
            vote_detail={"score": conf, "threshold": 0.5},
            vetoes=[]
        )
