"""Source-fusion utilities for topic-aware sentiment.

This module performs a reliability-weighted blend of per-source, per-topic
sentiment outputs into a single feature vector per symbol/topic.
"""
from typing import Dict, Any
import math

# Default source reliabilities (user editable in config)
DEFAULT_SOURCE_WEIGHTS = {
    "influencer": 3.0,  # high-trust influencer feeds
    "twitter": 1.5,
    "reddit": 1.0,
    "telegram": 0.8,
    "news": 0.7,
}


class SentimentFusion:
    def __init__(self, source_weights: Dict[str, float] = None):
        self.source_weights = source_weights or DEFAULT_SOURCE_WEIGHTS

    def fuse(self, per_source_topic_scores: Dict[str, Dict[str, float]],
             engagement: Dict[str, float] = None) -> Dict[str, Dict[str, float]]:
        """Aggregate per-source topic scores.

        per_source_topic_scores: {source: {topic: score_in_[-1,1]}}
        engagement: optional per-source scalar to boost active sources.

        Returns: {topic: {score: float, z: float, pctl: float}}
        """
        engagement = engagement or {}
        topic_acc = {}
        weight_acc = {}
        for source, topics in per_source_topic_scores.items():
            w = float(self.source_weights.get(source, 1.0))
            w *= float(engagement.get(source, 1.0))
            for topic, score in topics.items():
                topic_acc[topic] = topic_acc.get(topic, 0.0) + score * w
                weight_acc[topic] = weight_acc.get(topic, 0.0) + w

        out = {}
        for topic, s in topic_acc.items():
            w = weight_acc.get(topic, 1.0)
            avg = s / w
            # produce short/medium same for now, and simple z/pctl placeholders
            z = avg / (0.2 if avg != 0 else 1e-6)  # assume 0.2 stdev baseline
            pctl = 50 + (math.tanh(z) * 50)
            out[topic] = {
                "score_s": avg, "score_m": avg, "z": z, "pctl": pctl
            }
        return out
