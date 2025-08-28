"""Augmento connector scaffold (optional provider). Graceful fallback to local model.

This adapter maps provider responses into our internal topic taxonomy and
outputs the same per-source topic score dict expected by the fusion layer.

This is a minimal, testable scaffold. Add real HTTP client logic when API creds
are available; keep it optional so the SentimentEngine uses the local classifier
by default.
"""
from typing import Dict, Any, Optional


class AugmentoClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def is_available(self) -> bool:
        return bool(self.api_key)

    def map_to_topics(self, text: str) -> Dict[str, float]:
        """Placeholder mapping. In real implementation, call the provider and
        translate provider topic taxonomy into our TOPIC_TAXONOMY keys.
        Returns: {topic: score_in[-1,1]}
        """
        # Fallback: empty dict -> engine will use local classifier
        return {}


def provider_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("augmento_api_key"))
