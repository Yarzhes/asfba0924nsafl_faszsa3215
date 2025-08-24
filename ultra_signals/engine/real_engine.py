from typing import Dict, Optional

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


class RealSignalEngine:
    """A realistic signal engine that uses the scoring and ensemble logic."""

    def __init__(self, settings: Dict, feature_store: FeatureStore):
        self.settings = settings
        self.feature_store = feature_store

    # ---------- robust feature fetch ----------
    def _get_features_robust(self, symbol: str, ts_like) -> Optional[dict]:
        """
        Resolve common mismatches:
        - tz-aware vs tz-naive
        - bar-close rounding
        - index dtype (pd.Timestamp vs epoch seconds/ms/ns)
        - engine off-by-one bar (use <= ts fallback)
        """
        import math

        tf = (self.settings.get("runtime", {}) or {}).get("primary_timeframe", "5m")
        freq = _tf_to_pandas_freq(tf)

        # Parse incoming ts
        try:
            ts = pd.Timestamp(ts_like)
        except Exception:
            ts = pd.Timestamp(ts_like, unit="s") if isinstance(ts_like, (int, float)) else pd.Timestamp.utcnow()

        # Normalize to tz-naive (FeatureStore commonly stores naive)
        ts = ts.tz_localize(None) if ts.tzinfo is not None else ts

        # Candidate exact keys to try (Timestamp + epoch s/ms/ns), using lowercase 's'
        ts_s = int(ts.value // 10**9)
        ts_ms = int(ts.value // 10**6)
        ts_ns = int(ts.value)  # pandas int nanoseconds

        exact_candidates = [ts, ts.floor("s"), ts.floor("min"), ts_s, ts_ms, ts_ns]
        # Also try timeframe-aligned floor if we know the freq (e.g., '5min')
        if freq:
            try:
                exact_candidates.insert(1, ts.floor(freq))
            except Exception:
                pass

        # 1) Direct tries (also with nearest=True if supported)
        for cand in exact_candidates:
            try:
                feat = self.feature_store.get_features(symbol, cand)
                if feat:
                    logger.debug("Feature hit (exact) for {} at {} using key {}", symbol, ts, cand)
                    return feat
            except TypeError:
                # Some stores accept nearest=True
                try:
                    feat = self.feature_store.get_features(symbol, cand, nearest=True)
                    if feat:
                        logger.debug("Feature hit (exact+nearest) for {} at {} using key {}", symbol, ts, cand)
                        return feat
                except Exception:
                    pass
            except Exception:
                pass

        # 2) If the store exposes the raw table or an accessor, search <= ts
        series_like = None
        # Common internal layouts:
        for attr in ("_store", "_cache", "table", "data", "features"):
            obj = getattr(self.feature_store, attr, None)
            if obj is None:
                continue
            try:
                series_like = obj.get(symbol, None)
            except Exception:
                continue
            if series_like is not None:
                break

        if series_like is not None:
            # Obtain index keys
            try:
                idx = getattr(series_like, "index", None)
                if idx is None and hasattr(series_like, "keys"):
                    idx = list(series_like.keys())
            except Exception:
                idx = None

            if idx is not None:
                try:
                    # Convert arbitrary key -> nanoseconds since epoch for comparison
                    def to_ns(v):
                        try:
                            # ints: decide s/ms/ns by digits
                            if isinstance(v, int):
                                if v == 0:
                                    return 0
                                l = int(math.log10(abs(v)) + 1)
                                if l <= 10:      # seconds
                                    return v * 10**9
                                elif l <= 13:    # millis
                                    return v * 10**6
                                else:            # assume already ns
                                    return v
                            # pandas/py timestamps or strings:
                            return pd.Timestamp(v).value
                        except Exception:
                            return None

                    target_ns = ts.value
                    pairs = []
                    keys_iter = list(idx) if hasattr(idx, "__iter__") else [idx]
                    for k in keys_iter:
                        kns = to_ns(k)
                        if kns is not None:
                            pairs.append((kns, k))

                    # pick the largest key <= target
                    le_pairs = [pair for pair in pairs if pair[0] <= target_ns]
                    if le_pairs:
                        best_kns, best_key = max(le_pairs, key=lambda x: x[0])
                        try:
                            feat = self.feature_store.get_features(symbol, best_key)
                        except TypeError:
                            feat = None
                            try:
                                feat = self.feature_store.get_features(symbol, best_key, nearest=True)
                            except Exception:
                                pass
                        if feat:
                            delta_sec = (target_ns - best_kns) / 1e9
                            logger.debug(
                                "Feature hit (search<=) for {} at {} matched key {} (Δ={:.3f}s)",
                                symbol, ts, best_key, delta_sec
                            )
                            return feat
                except Exception:
                    pass

        # 3) Named helpers if provided
        for name in ("get_features_at_or_before", "get_features_nearest", "get_latest_features"):
            helper = getattr(self.feature_store, name, None)
            if helper is None:
                continue
            try:
                feat = None
                try:
                    feat = helper(symbol, ts)
                except TypeError:
                    # maybe expects integer or symbol-only
                    try:
                        feat = helper(symbol, ts_s)
                    except Exception:
                        feat = helper(symbol)
                if feat:
                    logger.debug("Feature hit via {} for {} at {}", name, symbol, ts)
                    return feat
            except Exception:
                continue

        return None
    # -----------------------------------------

    def generate_signal(
        self, ohlcv_segment: pd.DataFrame, symbol: str
    ) -> Optional[EnsembleDecision]:
        """Generates a signal for the given bar by building a full feature vector."""
        latest_bar = ohlcv_segment.iloc[-1]
        timestamp = latest_bar.name
        tf = self.settings["runtime"]["primary_timeframe"]
        ts_epoch = int(pd.Timestamp(timestamp).timestamp())

        # 1) Robustly fetch features for this bar
        features = self._get_features_robust(symbol, timestamp)
        if not features:
            logger.debug("Not enough data to compute features for {} at {}", symbol, timestamp)
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
        component_scores = scoring.component_scores(feature_vector, self.settings["features"])
        logger.debug("Component scores: {}", component_scores)
        
        # 4) Subsignals from scores
        subsignals = []
        for name, score in component_scores.items():
            if pd.isna(score) or score == 0:
                continue
            direction = "LONG" if score > 0 else "SHORT"
            subsignals.append(
                SubSignal(
                    ts=ts_epoch,
                    symbol=symbol,
                    tf=tf,
                    strategy_id=name,
                    direction=direction,
                    confidence_raw=abs(score),
                    confidence_calibrated=abs(score),  # no calibration yet
                    reasons={},
                )
            )

        if not subsignals:
            logger.debug("No subsignals generated; returning explicit FLAT.")
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

        # 5) Regime
        current_regime = regime.detect_regime(feature_vector, self.settings["regime"])
        logger.debug("Detected regime: {}", current_regime)

        # 6) Combine
        try:
            logger.debug(
                "Subsignals: {}",
                [(s.strategy_id, getattr(s, "confidence_calibrated", None), getattr(s, "direction", None)) for s in subsignals]
            )
        except Exception:
            logger.debug("Subsignals present but not printable")

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
        px = bar["close"]
        atr = getattr(features.get("volatility"), "atr", None) or 0
        # Defaults if not already on the position/config
        k_stop = getattr(pos, "atr_mult_stop", 2.0)
        k_tp   = getattr(pos, "atr_mult_tp",   3.0)
        max_bars = getattr(pos, "max_bars", 288)  # e.g., 24h on 5m bars

        if pos.side == "LONG":
            if atr and px <= pos.entry_price - k_stop*atr: return "STOP"
            if atr and px >= pos.entry_price + k_tp*atr:   return "TP"
        else:  # SHORT
            if atr and px >= pos.entry_price + k_stop*atr: return "STOP"
            if atr and px <= pos.entry_price - k_tp*atr:   return "TP"

        if pos.bars_held >= max_bars:
            return "TIME_STOP"
        return None
