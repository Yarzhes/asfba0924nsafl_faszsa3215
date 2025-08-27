from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from loguru import logger

# NOTE: We avoid heavy imports (snscrape, requests) until actually used to keep startup light.

@dataclass
class BaseCollector:
    kind: str
    cfg: Dict[str, Any]
    root_cfg: Dict[str, Any]
    cache_dir: Path
    _last_run: float = 0.0

    def refresh_interval(self) -> int:
        return int(self.cfg.get("refresh_sec", 900))

    def should_refresh(self, now: float) -> bool:
        return (now - self._last_run) >= self.refresh_interval()

    def collect(self) -> List[Dict[str, Any]]:  # pragma: no cover
        self._last_run = time.time()
        return []

    # Helper for mapping tickers/keywords to symbols via config map
    def _map_symbols(self, text: str) -> List[str]:
        sym_map = (self.root_cfg.get("symbol_keyword_map") or {}) if isinstance(self.root_cfg.get("symbol_keyword_map"), dict) else {}
        out = []
        lower = text.lower()
        for sym, kws in sym_map.items():
            for kw in kws:
                if kw.lower() in lower:
                    out.append(sym)
                    break
        return out

# --------------------------- Twitter via snscrape ---------------------------
class TwitterCollector(BaseCollector):
    def collect(self) -> List[Dict[str, Any]]:  # pragma: no cover (network)
        super().collect()
        items: List[Dict[str, Any]] = []
        try:
            import subprocess, json, shlex
            influencers = list((self.root_cfg.get("influencers") or {}).keys())
            max_items = int(self.cfg.get("max_items", 150))
            for handle in influencers[:10]:  # simple cap
                # Use snscrape CLI (fallback approach). This can be swapped for python API later.
                cmd = f"snscrape --max-results {max_items} twitter-user {handle}"
                try:
                    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=30)
                    if proc.returncode != 0:
                        continue
                    lines = proc.stdout.strip().splitlines()
                    for ln in lines:
                        if not ln.strip():
                            continue
                        # snscrape returns JSON lines by default
                        try:
                            obj = json.loads(ln)
                        except Exception:
                            continue
                        txt = obj.get("content") or ""
                        syms = self._map_symbols(txt)
                        if not syms:
                            continue
                        items.append({
                            "ts": int(time.time()),  # fallback; could parse obj['date']
                            "text": txt,
                            "symbols": syms,
                            "meta": {"author": handle, "likes": obj.get("likeCount"), "retweets": obj.get("retweetCount")},
                        })
                except Exception:
                    continue
        except Exception:
            logger.debug("snscrape not available; TwitterCollector returning empty set")
        return items

# --------------------------- Reddit JSON Collector ---------------------------
class RedditCollector(BaseCollector):
    def collect(self) -> List[Dict[str, Any]]:  # pragma: no cover (network)
        super().collect()
        items: List[Dict[str, Any]] = []
        subs = self.root_cfg.get("subreddit_list") or []
        import json, urllib.request
        headers = {"User-Agent": "UltraSignalsSentimentBot/0.1"}
        for sub in subs[:10]:
            url = f"https://www.reddit.com/r/{sub}/new.json?limit=50"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8", "ignore"))
                for child in data.get("data", {}).get("children", []):
                    d = child.get("data", {})
                    txt = (d.get("title") or "") + " \n" + (d.get("selftext") or "")
                    syms = self._map_symbols(txt)
                    if not syms:
                        continue
                    items.append({
                        "ts": int(d.get("created_utc", time.time())),
                        "text": txt,
                        "symbols": syms,
                        "meta": {"sub": sub, "score": d.get("score")},
                    })
            except Exception:
                continue
        return items

# --------------------------- Fear & Greed Index ---------------------------
class FearGreedCollector(BaseCollector):
    def collect(self) -> List[Dict[str, Any]]:  # pragma: no cover (network)
        super().collect()
        items: List[Dict[str, Any]] = []
        import json, urllib.request
        url = "https://api.alternative.me/fng/"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8", "ignore"))
            value = float(data.get("data", [{}])[0].get("value", 0))
            items.append({
                "ts": int(time.time()),
                "text": f"FGI={value}",
                "symbols": [s for s in (self.root_cfg.get("symbols") or [])],
                "meta": {"fg_index": value},
            })
        except Exception:
            pass
        return items

# --------------------------- Funding / OI (positioning sentiment) -----------
class FundingCollector(BaseCollector):
    def collect(self) -> List[Dict[str, Any]]:  # pragma: no cover (network)
        super().collect()
        items: List[Dict[str, Any]] = []
        # Simple rolling caches (module-level dicts) for funding & oi history
        cache_key = "_funding_cache"
        if not hasattr(FundingCollector, cache_key):  # type: ignore[attr-defined]
            setattr(FundingCollector, cache_key, {})  # type: ignore[attr-defined]
        fcache = getattr(FundingCollector, cache_key)  # type: ignore[attr-defined]
        # Simulate pulling funding & oi from public endpoints (placeholder deterministic values)
        now = int(time.time())
        for i, sym in enumerate((self.root_cfg.get("symbols") or [])[:30]):
            rec = fcache.setdefault(sym, {"funding": [], "oi": []})
            # Mock values with mild variation
            val_f = ((now // 3600) % 24 - 12) / 1000.0 + (i * 0.0001)
            val_oi = 1000 + ((now // 60) % 120) * 2 + i * 5
            rec["funding"].append(val_f)
            rec["oi"].append(val_oi)
            rec["funding"] = rec["funding"][-200:]
            rec["oi"] = rec["oi"][-200:]
            # z-score helpers
            def _z(series):
                if len(series) < 30:
                    return 0.0
                import statistics as stats
                mu = stats.mean(series)
                try:
                    sd = stats.pstdev(series)
                except Exception:
                    sd = 0.0
                if sd <= 1e-9:
                    return 0.0
                return (series[-1] - mu) / sd
            funding_z = _z(rec["funding"])
            oi_z = _z(rec["oi"])
            items.append({
                "ts": now,
                "text": f"funding {val_f:+.4f} oi {val_oi}",
                "symbols": [sym],
                "meta": {"funding": val_f, "funding_z": funding_z, "oi_z": oi_z},
            })
        return items
