"""Simple rule-based topic classifier with lightweight probabilistic output.

This module provides a no-key, local fallback classifier that uses regex seeds
from the taxonomy. It is intentionally compact and deterministic so it can be
used in CI/backtest without heavy dependencies.
"""
from typing import Dict, List, Tuple
import re
import math
from .topics import TOPIC_TAXONOMY, DEFAULT_TOPIC


class TopicClassifier:
    def __init__(self, taxonomy: Dict = None):
        self.taxonomy = taxonomy or TOPIC_TAXONOMY
        # precompile patterns
        self._patterns = {}
        for k, v in self.taxonomy.items():
            self._patterns[k] = [re.compile(p, re.IGNORECASE) for p in v.get("seeds", [])]

    def classify(self, text: str) -> Dict[str, float]:
        """Return a probability-like dict over topics for the given text.

        The output is normalized to sum to 1. If no topic matches, returns
        {DEFAULT_TOPIC: 1.0}.
        """
        if not text:
            return {DEFAULT_TOPIC: 1.0}

        scores = {k: 0.0 for k in self.taxonomy.keys()}
        for topic, patterns in self._patterns.items():
            for p in patterns:
                if p.search(text):
                    scores[topic] += 1.0

        total = sum(scores.values())
        if total == 0:
            return {DEFAULT_TOPIC: 1.0}

        # soft-normalize with sqrt to dampen outliers
        norm_scores = {k: math.sqrt(v) for k, v in scores.items() if v > 0}
        s = sum(norm_scores.values())
        return {k: v / s for k, v in norm_scores.items()}


def extract_symbols(text: str) -> List[str]:
    """Very small heuristic to map hashtags/tickers to symbols like BTC, ETH.

    Returns a list of uppercase ticker strings found in the text. This is
    intentionally conservative and returns short strings of 2-6 uppercase
    letters/numbers.
    """
    if not text:
        return []
    # $BTC or #BTC or BTC/USDT or BTC
    candidates = set()
    # $TICKER style
    for m in re.findall(r"\$([A-Za-z]{2,6})", text):
        candidates.add(m.upper())
    # #hashtag
    for m in re.findall(r"#([A-Za-z]{2,6})", text):
        candidates.add(m.upper())
    # bare token uppercase (word boundaries)
    for m in re.findall(r"\b([A-Z]{2,6})\b", text):
        candidates.add(m.upper())
    return list(candidates)
