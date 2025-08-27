from __future__ import annotations

import re
from typing import Dict, Any

EMOJI_MAP = {
    "🚀": 0.8,
    "🔥": 0.6,
    "💎": 0.7,
    "😱": -0.6,
    "😭": -0.7,
    "🩸": -0.6,
}

LEXICON = {
    # bullish
    "bull": 0.5, "moon": 0.7, "pump": 0.6, "breakout": 0.4, "rally": 0.5,
    # bearish
    "dump": -0.6, "crash": -0.8, "rekt": -0.7, "bear": -0.4, "rug": -0.8,
    # neutral / risk terms (mild adjustments)
    "volatility": -0.05, "uncertain": -0.1, "fear": -0.3, "greed": 0.1,
}

WORD_RE = re.compile(r"[a-zA-Z#]+")

class SentimentScorer:
    """Lightweight lexicon + emoji polarity scorer.

    Optional: can be extended with a transformer model (HF, local only) by
    overriding _model_score(). We keep it deterministic & fast for now.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.model_name = self.cfg.get("transformer_model")

    def score_text(self, text: str, meta: Dict[str, Any] | None = None) -> Dict[str, float]:
        if not text:
            return {"polarity": 0.0, "confidence": 0.0}
        txt = text.lower()
        score = 0.0
        weight_total = 0.0
        for m in WORD_RE.findall(txt):
            base = LEXICON.get(m.strip("#"))
            if base is not None:
                score += base
                weight_total += 1.0
        for ch, val in EMOJI_MAP.items():
            if ch in text:
                score += val
                weight_total += 1.0
        if weight_total > 0:
            score /= weight_total
        # Clamp
        if score > 1:
            score = 1.0
        if score < -1:
            score = -1.0
        return {"polarity": float(score), "confidence": float(min(1.0, abs(score))) }
