from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from loguru import logger

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from .scoring import SentimentScorer
from .aggregator import SentimentAggregator
from .sources import BaseCollector, TwitterCollector, RedditCollector, FearGreedCollector, FundingCollector


@dataclass
class SentimentSnapshot:
    ts: int
    symbol: str
    scores: Dict[str, float]  # e.g. {'sent_score_s': 0.12, 'sent_score_m': 0.34, 'infl_weighted': 0.20}
    flags: Dict[str, int]     # e.g. {'extreme_bull':0,'extreme_bear':1}
    raw: Dict[str, Any] = field(default_factory=dict)


class SentimentEngine:
    """High-level orchestrator for the unified sentiment pipeline.

    Responsibilities:
      * Manage source collectors (Twitter, Reddit, Funding, Fear & Greed, etc.)
      * Run scoring on newly collected textual items
      * Maintain rolling aggregation windows per symbol
      * Emit structured feature dicts for downstream model / veto layer

    This is a lean MVP skeleton; individual collectors can be fleshed out
    incrementally without changing the public interface.
    """

    def __init__(self, settings: Dict[str, Any], feature_store: Any | None = None):
        self.settings = settings or {}
        sent_cfg = (self.settings.get("sentiment") or {}) if isinstance(self.settings.get("sentiment"), dict) else {}
        self.enabled = bool(sent_cfg.get("enabled", True))
        self.cache_dir = Path(sent_cfg.get("cache_dir", ".cache/sentiment"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.symbols = sent_cfg.get("symbols") or (self.settings.get("runtime", {}) or {}).get("symbols", [])
        self.scorer = SentimentScorer(sent_cfg)
        self.aggregator = SentimentAggregator(sent_cfg)
        self.collectors: List[BaseCollector] = []
        self._fs = feature_store  # optional FeatureStore to persist snapshots
        self._last_metrics_export: float = 0.0
        self._telemetry_cfg = sent_cfg.get("telemetry", {}) if isinstance(sent_cfg.get("telemetry"), dict) else {}
        if not self.enabled:
            logger.warning("SentimentEngine initialized but disabled via config.")
            return
        self._init_collectors(sent_cfg)
        logger.info("SentimentEngine ready (symbols=%s collectors=%d)", self.symbols, len(self.collectors))

    # ------------------------------------------------------------------
    def _init_collectors(self, cfg: Dict[str, Any]):
        src_cfg = cfg.get("sources", {})
        def _add(kind: str, cls):
            c = src_cfg.get(kind, {})
            if not c or not c.get("enabled", True):
                return
            try:
                self.collectors.append(cls(kind, c, cfg, self.cache_dir))
            except Exception as e:  # pragma: no cover (hard-fail resilience)
                logger.exception("Failed to init collector %s: %s", kind, e)
        _add("twitter", TwitterCollector)
        _add("reddit", RedditCollector)
        _add("fear_greed", FearGreedCollector)
        _add("funding", FundingCollector)
        # Optional future: discord, trends, news

    # ------------------------------------------------------------------
    def poll_sources(self) -> List[Dict[str, Any]]:
        """Poll all collectors that indicate they are stale.
        Returns list of raw items: each item is a dict with at minimum keys:
          ts (epoch sec), text (str), symbols (list[str]), meta (dict)
        """
        items: List[Dict[str, Any]] = []
        now = time.time()
        for col in self.collectors:
            try:
                if col.should_refresh(now):
                    new_items = col.collect()
                    if new_items:
                        items.extend(new_items)
            except Exception as e:  # pragma: no cover
                logger.exception("Collector %s failed: %s", col.kind, e)
        return items

    # ------------------------------------------------------------------
    def step(self) -> Dict[str, SentimentSnapshot]:
        """Run one pipeline iteration: poll sources, score, aggregate, detect extremes.
        Returns per-symbol snapshot mapping.
        """
        if not self.enabled:
            return {}
        raw_items = self.poll_sources()
        if not raw_items:
            # Still push existing latest into FeatureStore & maybe export metrics
            if self._fs is not None:
                for sym, snap in self.aggregator.latest_per_symbol.items():
                    try:
                        self._fs.set_sentiment_snapshot(sym, snap)
                    except Exception:
                        pass
            self._maybe_export_metrics()
            return {}
        # Score each textual item
        for it in raw_items:
            if not it.get("text"):
                continue
            try:
                sc = self.scorer.score_text(it["text"], meta=it.get("meta"))
                it["polarity"] = sc["polarity"]
                it["polarity_conf"] = sc.get("confidence")
            except Exception:
                it["polarity"] = 0.0
        # Feed into aggregator
        agg_results = self.aggregator.ingest(raw_items)
        snapshots: Dict[str, SentimentSnapshot] = {}
        for sym, res in agg_results.items():
            snap = SentimentSnapshot(
                ts=res.get("ts", int(time.time())),
                symbol=sym,
                scores={k: v for k, v in res.items() if k.startswith("sent_") or k.endswith("_z") or k in {"infl_weighted","funding_z","oi_z","basis_z","fg_index"}},
                flags={k: int(v) for k, v in res.items() if k.startswith("extreme_") or k.endswith("_flag")},
                raw={k: v for k, v in res.items() if k not in ("ts",)}
            )
            snapshots[sym] = snap
            # Persist into FeatureStore if provided
            if self._fs is not None:
                try:
                    self._fs.set_sentiment_snapshot(sym, snap.scores | snap.flags)
                except Exception:
                    pass
        self._maybe_export_metrics()
        return snapshots

    # ------------------------------------------------------------------
    def feature_view(self) -> Dict[str, Dict[str, float]]:
        """Return latest aggregated sentiment feature view (lightweight dict)."""
        out: Dict[str, Dict[str, float]] = {}
        for sym, snap in self.aggregator.latest_per_symbol.items():
            out[sym] = snap.copy()
        return out

    # ------------------------------------------------------------------
    def maybe_veto(self, symbol: str) -> Optional[str]:
        """Return veto reason if sentiment extremes should block a trade."""
        cfg = (self.settings.get("sentiment") or {})
        if not cfg or not cfg.get("veto_extremes", True):
            return None
        latest = self.aggregator.latest_per_symbol.get(symbol)
        if not latest:
            return None
        if latest.get("extreme_flag_bull"):
            return "SENTIMENT_EUPHORIA"
        if latest.get("extreme_flag_bear"):
            return "SENTIMENT_PANIC"
        return None

    def size_modifier(self, symbol: str) -> float:
        cfg = (self.settings.get("sentiment") or {})
        if not cfg.get("size_dampen_extremes", True):
            return 1.0
        latest = self.aggregator.latest_per_symbol.get(symbol)
        if not latest:
            return 1.0
        factor = float(cfg.get("size_dampen_factor", 0.5))
        if latest.get("extreme_flag_bull") or latest.get("extreme_flag_bear"):
            return factor
        return 1.0

    # ------------------------------------------------------------------
    def _maybe_export_metrics(self) -> None:
        try:
            if not self._telemetry_cfg.get("emit_metrics", True):
                return
            interval = int(self._telemetry_cfg.get("interval_sec", 300))
            now = time.time()
            if (now - self._last_metrics_export) < interval:
                return
            path = Path(self._telemetry_cfg.get("export_path", "sentiment_metrics.csv"))
            rows = []
            for sym, snap in self.aggregator.latest_per_symbol.items():
                row = {"symbol": sym, **snap}
                rows.append(row)
            if not rows:
                return
            import csv
            write_header = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
                if write_header:
                    w.writeheader()
                for r in rows:
                    w.writerow(r)
            self._last_metrics_export = now
        except Exception:
            pass
