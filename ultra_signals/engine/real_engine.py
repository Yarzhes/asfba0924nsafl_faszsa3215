from typing import Dict, Optional, Any, List

import pandas as pd
from loguru import logger

from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import EnsembleDecision, FeatureVector, SubSignal
from ultra_signals.engine import ensemble, regime, scoring


def _trace_engine_flat(symbol: str, timeframe: str, ts_epoch: int,
                       decision: Optional[EnsembleDecision], reason: str) -> None:
    """Debug helper so FLAT reasons show up next to EventRunner logs."""
    try:
        vd = getattr(decision, "vote_detail", {}) if decision is not None else {}
        vetoes = getattr(decision, "vetoes", []) if decision is not None else []
        conf = float(getattr(decision, "confidence", 0.0) or 0.0) if decision is not None else 0.0
        logger.debug(
            "[ENGINE] FLAT explain symbol={} tf={} ts={} reason={} conf={:.3f} vote_detail={} vetoes={}",
            symbol, timeframe, ts_epoch, reason, conf, vd, vetoes
        )
    except Exception:
        pass


def _tf_to_pandas_freq(tf: str) -> Optional[str]:
    """
    Convert common timeframe strings ('1m', '5m', '1h', '1d') into pandas
    offset aliases used by floor() / ceil(). If unknown, return None.
    """
    if not tf:
        return None
    tf = str(tf).lower().strip()
    if tf.endswith("m"):
        try:
            n = int(tf[:-1])
            return f"{n}min"
        except Exception:
            return None
    if tf.endswith("h"):
        try:
            n = int(tf[:-1])
            return f"{n}H"
        except Exception:
            return None
    if tf.endswith("d"):
        try:
            n = int(tf[:-1])
            return f"{n}D"
        except Exception:
            return None
    return None


# ---- NEW: tiny helper to dump store internals when features are missing ----
def _log_store_state_for_debug(fs: FeatureStore, symbol: str, timeframe: Optional[str]) -> None:
    """
    Best-effort peek at the FeatureStore so missing-feature cases are self-explanatory
    in logs. Wrapped in try/except so it never breaks runtime even if internals change.
    """
    try:
        store_id = id(fs)
        known_tfs = []
        sym_cache_keys = []
        latest_ts = None
        try:
            cache = getattr(fs, "_features", {})
            sym_cache = cache.get(symbol, {})
            # sym_cache is typically {tf: {ts: features}} or similar; we only list keys
            known_tfs = list(sym_cache.keys())
            sym_cache_keys = known_tfs
        except Exception:
            pass
        try:
            latest = fs.get_latest_features(symbol, timeframe)
            if isinstance(latest, dict):
                latest_ts = latest.get("ts") or latest.get("timestamp") or None
        except Exception:
            pass
        logger.debug(
            "[ENGINE] Missing features debug: store_id={} symbol={} requested_tf={} known_timeframes={} latest_ts={}",
            store_id, symbol, timeframe, sym_cache_keys, latest_ts
        )
    except Exception:
        # never let debug helper throw
        pass
# ---------------------------------------------------------------------------


class RealSignalEngine:
    """A realistic signal engine that uses the scoring and ensemble logic."""

    def __init__(self, settings: Dict, feature_store: FeatureStore):
        self.settings = settings or {}
        self.feature_store = feature_store

    # ---------- robust feature fetch ----------
    def _get_features_robust(self, symbol: str, timestamp_like: Any, timeframe: Optional[str]) -> Optional[dict]:
        """
        Resolve common mismatches:
        - tz-aware vs tz-naive
        - bar-close rounding (floor to timeframe)
        - epoch s/ms/ns inputs
        - off-by-one bar -> use at-or-before fallback
        """
        # Normalize ts to tz-naive Timestamp
        try:
            ts = pd.Timestamp(timestamp_like)
        except Exception:
            if isinstance(timestamp_like, (int, float)):
                ts = pd.to_datetime(int(timestamp_like), unit="s", utc=False)
            else:
                ts = pd.Timestamp.utcnow()
        if ts.tzinfo is not None:
            try:
                ts = ts.tz_convert(None)
            except Exception:
                ts = ts.tz_localize(None)

        # 1) Preferred: timeframe-aware call (FeatureStore accepts both styles)
        try:
            feats = self.feature_store.get_features(symbol, timeframe, ts, nearest=True)
            if feats:
                logger.debug("Feature hit via timeframe-aware get_features for {} {} at {}", symbol, timeframe, ts)
                return feats
        except TypeError:
            pass
        except Exception:
            pass

        # 2) Legacy form (search across tfs)
        try:
            feats = self.feature_store.get_features(symbol, ts, nearest=True)
            if feats:
                logger.debug("Feature hit via legacy get_features for {} at {}", symbol, ts)
                return feats
        except Exception:
            pass

        # 3) Last resort: latest available features for this symbol (or timeframe if provided)
        try:
            latest = self.feature_store.get_latest_features(symbol, timeframe)
            if latest:
                logger.debug("Falling back to latest features for {} {}", symbol, timeframe or "")
                return latest
        except Exception:
            pass

        return None
    # -----------------------------------------

    def generate_signal(
        self, ohlcv_segment: pd.DataFrame, symbol: str
    ) -> Optional[EnsembleDecision]:
        """Generates a signal for the given bar by building a full feature vector."""
        if ohlcv_segment is None or len(ohlcv_segment.index) == 0:
            ts_epoch = 0
            tf = str((self.settings.get("runtime", {}) or {}).get("primary_timeframe", "5m"))
            dec = EnsembleDecision(
                ts=ts_epoch,
                symbol=symbol,
                tf=tf,
                decision="FLAT",
                confidence=0.0,
                subsignals=[],
                vote_detail={"reason": "no_data"},
                vetoes=[],
            )
            _trace_engine_flat(symbol, tf, ts_epoch, dec, "no_data")
            return dec

        latest_bar = ohlcv_segment.iloc[-1]
        timestamp = latest_bar.name
        tf = str((self.settings.get("runtime", {}) or {}).get("primary_timeframe", "5m"))
        ts_epoch = int(pd.Timestamp(timestamp).timestamp())

        # Optional warmup guard: block until N bars exist (only if configured)
        min_hist = int(((self.settings.get("engine", {}) or {}).get("min_history_bars", 0)) or 0)
        if min_hist > 0:
            try:
                have = int(self.feature_store.get_warmup_status(symbol, tf))
            except Exception:
                have = 0
            if have < min_hist:
                dec = EnsembleDecision(
                    ts=ts_epoch,
                    symbol=symbol,
                    tf=tf,
                    decision="FLAT",
                    confidence=0.0,
                    subsignals=[],
                    vote_detail={"reason": "warmup"},
                    vetoes=[],
                )
                _trace_engine_flat(symbol, tf, ts_epoch, dec, "warmup")
                return dec

        # 1) Robustly fetch features for this bar (timeframe-aware)
        features = self._get_features_robust(symbol, timestamp, tf)
        if not features:
            # ---- NEW: richer debug so it's obvious when a different FeatureStore is being read ----
            try:
                logger.debug(
                    "Not enough data to compute features for {} at {} (store_id={})",
                    symbol, timestamp, id(self.feature_store)
                )
                _log_store_state_for_debug(self.feature_store, symbol, tf)
            except Exception:
                pass
            # ---------------------------------------------------------------------------------------

            dec = EnsembleDecision(
                ts=ts_epoch,
                symbol=symbol,
                tf=tf,
                decision="FLAT",
                confidence=0.0,
                subsignals=[],
                vote_detail={"reason": "insufficient_features"},
                vetoes=[],
            )
            _trace_engine_flat(symbol, tf, ts_epoch, dec, "insufficient_features")
            return dec

        # 2) Full feature vector
        feature_vector = FeatureVector(
            symbol=symbol,
            timeframe=tf,
            ohlcv=latest_bar.to_dict(),
            trend=features.get("trend"),
            momentum=features.get("momentum"),
            volatility=features.get("volatility"),
            volume_flow=features.get("volume_flow"),
            derivatives=None,
            orderbook=None,
            rs=None,
        )
        try:
            logger.debug("FV for {} at {}:\n{}", symbol, timestamp, feature_vector.model_dump_json(indent=2))
        except Exception:
            logger.debug("FV for {} at {} present (not printable).", symbol, timestamp)

        # 3) Component scores
        comp_cfg = (self.settings.get("features") or {})
        component_scores = scoring.component_scores(feature_vector, comp_cfg)
        logger.debug("Component scores: {}", component_scores)

        # 4) Subsignals from scores
        subsignals: List[SubSignal] = []
        for name, score in component_scores.items():
            try:
                val = float(score)
            except Exception:
                continue
            if pd.isna(val) or val == 0.0:
                continue
            direction = "LONG" if val > 0.0 else "SHORT"
            subsignals.append(
                SubSignal(
                    ts=ts_epoch,
                    symbol=symbol,
                    tf=tf,
                    strategy_id=name,
                    direction=direction,
                    confidence_raw=abs(val),
                    confidence_calibrated=abs(val),
                    reasons={},
                )
            )

        if not subsignals:
            dec = EnsembleDecision(
                ts=ts_epoch,
                symbol=symbol,
                tf=tf,
                decision="FLAT",
                confidence=0.0,
                subsignals=[],
                vote_detail={"reason": "no_subsignals"},
                vetoes=[],
            )
            _trace_engine_flat(symbol, tf, ts_epoch, dec, "no_subsignals")
            return dec

        # 5) Regime & 6) Combine
        current_regime = regime.detect_regime(feature_vector, (self.settings.get("regime") or {}))
        logger.debug("Detected regime: {}", current_regime)

        final_decision = ensemble.combine_subsignals(subsignals, current_regime, self.settings)

        if final_decision is None:
            dec = EnsembleDecision(
                ts=ts_epoch,
                symbol=symbol,
                tf=tf,
                decision="FLAT",
                confidence=0.0,
                subsignals=subsignals,
                vote_detail={"reason": "combine_returned_none"},
                vetoes=[],
            )
            _trace_engine_flat(symbol, tf, ts_epoch, dec, "combine_returned_none")
            return dec

        if final_decision.decision != "FLAT":
            logger.info("Subsignals for {} at {}: {}", symbol, timestamp, subsignals)
            logger.info(
                "Final Decision: {} @ {} (Vote Detail: {})",
                final_decision.decision,
                final_decision.confidence,
                final_decision.vote_detail,
            )
        else:
            _trace_engine_flat(symbol, tf, ts_epoch, final_decision, "flat_after_combine")
            logger.debug("Final Decision for {} at {}: FLAT (Vote Detail: {})", symbol, timestamp, final_decision.vote_detail)

        return final_decision

    def should_exit(self, symbol, pos, bar, features):
        """ATR/Time-based exits (unchanged semantics, safer casting)."""
        try:
            px = float(bar["close"])
        except Exception:
            px = float(getattr(bar, "close", 0.0))

        vol = features.get("volatility") if isinstance(features, dict) else None
        atr = getattr(vol, "atr", None) if vol is not None else None
        try:
            atr = float(atr) if atr is not None else 0.0
        except Exception:
            atr = 0.0

        k_stop = float(getattr(pos, "atr_mult_stop", 2.0))
        k_tp   = float(getattr(pos, "atr_mult_tp",   3.0))
        max_bars = int(getattr(pos, "max_bars", 288))
        entry = float(getattr(pos, "entry_price", px))
        side = str(getattr(pos, "side", "LONG")).upper()

        if side == "LONG":
            if atr and px <= entry - k_stop * atr: return "STOP"
            if atr and px >= entry + k_tp * atr:   return "TP"
        else:
            if atr and px >= entry + k_stop * atr: return "STOP"
            if atr and px <= entry - k_tp * atr:   return "TP"

        if int(getattr(pos, "bars_held", 0)) >= max_bars:
            return "TIME_STOP"
        return None
