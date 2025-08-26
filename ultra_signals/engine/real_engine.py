from typing import Dict, Optional, Any, List

import pandas as pd
from loguru import logger

from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import EnsembleDecision, FeatureVector, SubSignal, RiskEvent
from ultra_signals.engine import ensemble, regime, scoring
from ultra_signals.risk.position_sizing import PositionSizing
from ultra_signals.engine.regime_router import RegimeRouter
from ultra_signals.engine.orderflow import OrderFlowAnalyzer, OrderFlowSnapshot, apply_orderflow_modulation
from ultra_signals.engine.liquidation_heatmap import LiquidationHeatmap
from ultra_signals.engine.position_sizer import PositionSizer
from ultra_signals.engine.execution_planner import select_playbook, build_plan
from ultra_signals.engine.news_veto import NewsVeto
from ultra_signals.engine.quality_gates import QualityGates


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
            flow_metrics=features.get("flow_metrics")
        )
        try:
            logger.debug("FV for {} at {}:\n{}", symbol, timestamp, feature_vector.model_dump_json(indent=2))
        except Exception:
            logger.debug("FV for {} at {} present (not printable).", symbol, timestamp)

        # 3) Component scores
        comp_cfg = (self.settings.get("features") or {})
        component_scores = scoring.component_scores(feature_vector, comp_cfg)
        logger.debug("Component scores: {}", component_scores)

        # 3b) (Sprint 14) Order Flow snapshot (mock-friendly). In a real system, you would pull
        # recent trades, liquidation events, and orderbook levels from data feeds / store.
        # Here we attempt to access optional FeatureStore hooks if they exist.
        orderflow_cfg = (self.settings.get("orderflow") or {}) if isinstance(self.settings, dict) else {}
        of_snapshot: OrderFlowSnapshot | None = None
        if orderflow_cfg.get("enable", True):
            try:
                trades = getattr(self.feature_store, "get_recent_trades", lambda *a, **k: [])(symbol, 50)
                liqs = getattr(self.feature_store, "get_recent_liquidations", lambda *a, **k: [])(symbol, 200)
                ob = getattr(self.feature_store, "get_orderbook_levels", lambda *a, **k: {})(symbol, 10)
                prev_cvd = getattr(self.feature_store, "get_prev_cvd", lambda *a, **k: None)(symbol)
                of_snapshot = OrderFlowAnalyzer.build_snapshot(trades, liqs, ob, self.settings, prev_cvd)
            except Exception:
                of_snapshot = None

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
                    confidence_calibrated=abs(val),
                    reasons={},
                )
            )

        # Sprint 11: create lightweight subsignals from flow metrics (cvd, oi_rate, liquidation pulse, depth imbalance)
        try:
            fm = features.get("flow_metrics") if isinstance(features, dict) else None
            fm_weights = (self.settings.get("weights_profiles") or {}).get("flow_metrics_raw", {})  # optional bucket
            if fm:
                # Map metric to (value, threshold default)
                fm_map = {
                    "cvd": getattr(fm, "cvd_chg", None) or getattr(fm, "cvd", None),
                    "oi_rate": getattr(fm, "oi_rate", None),
                    "liquidation_pulse": getattr(fm, "liq_cluster", None),
                    "depth_imbalance": getattr(fm, "depth_imbalance", None),
                }
                for sid, raw_val in fm_map.items():
                    if raw_val is None:
                        continue
                    try:
                        v = float(raw_val)
                    except Exception:
                        continue
                    if v == 0:
                        continue
                    direction = "LONG" if v > 0 else "SHORT"
                    conf = min(1.0, abs(v))
                    subsignals.append(SubSignal(
                        ts=ts_epoch,
                        symbol=symbol,
                        tf=tf,
                        strategy_id=sid,
                        direction=direction,
                        confidence_calibrated=conf,
                        reasons={"src": "flow_metrics"}
                    ))
        except Exception:
            pass

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

        # 5) Regime router (Sprint 13) - determines profile & active alphas
        router_info = RegimeRouter.route({
            "trend": features.get("trend"),
            "momentum": features.get("momentum"),
            "volatility": features.get("volatility"),
            "regime": features.get("regime"),
        }, self.settings)
        current_regime = router_info.get("regime", "mixed")
        active_alphas = set(router_info.get("alphas") or [])
        weight_scale = float(router_info.get("weight_scale", 1.0))
        min_conf_router = float(router_info.get("min_confidence", 0.0))

        # Down-weight or remove subsignals not in active alpha list if list not trivial
        filtered: List[SubSignal] = []
        for s in subsignals:
            sid = getattr(s, "strategy_id", "")
            if active_alphas and "none" not in active_alphas and active_alphas:
                if sid in active_alphas:
                    # scale confidence if above min_conf_router
                    if s.confidence_calibrated >= min_conf_router:
                        s.confidence_calibrated = min(1.0, s.confidence_calibrated * weight_scale)
                    filtered.append(s)
                else:
                    # skip non-active
                    continue
            else:
                filtered.append(s)
        subsignals = filtered

        if not subsignals:
            dec = EnsembleDecision(
                ts=ts_epoch,
                symbol=symbol,
                tf=tf,
                decision="FLAT",
                confidence=0.0,
                subsignals=[],
                vote_detail={"reason": "router_filtered_all", "regime_router": router_info},
                vetoes=[],
            )
            _trace_engine_flat(symbol, tf, ts_epoch, dec, "router_filtered_all")
            return dec

        final_decision = ensemble.combine_subsignals(subsignals, current_regime, self.settings)
        # Sprint 14: apply order flow confidence modulation post-ensemble but pre-risk sizing.
        if final_decision and of_snapshot and final_decision.decision in ("LONG", "SHORT"):
            try:
                new_conf, of_detail = apply_orderflow_modulation(final_decision.decision, final_decision.confidence, of_snapshot, orderflow_cfg)
                final_decision.confidence = new_conf
                final_decision.vote_detail.setdefault("orderflow", of_detail)
            except Exception:
                pass
        # Attach router telemetry
        try:
            final_decision.vote_detail.setdefault("regime_router", router_info)
        except Exception:
            pass

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
        # Attach meta-router profile metadata if present in settings
        try:
            mr = (self.settings or {}).get('meta_router')
            if mr:
                final_decision.vote_detail.setdefault('profile', mr)
                # Sprint 20 safety: if profile missing or stale, either downscale or veto
                if final_decision.decision in ("LONG", "SHORT"):
                    if mr.get('missing'):
                        # Fail-closed: veto trade
                        final_decision.vetoes.append('VETO_PROFILE_MISSING')
                        final_decision.decision = 'FLAT'
                        final_decision.vote_detail.setdefault('reason', 'PROFILE_MISSING')
                    elif mr.get('stale'):
                        # Apply size downgrade factor if present in settings
                        downgrade = float((self.settings.get('profiles', {}) or {}).get('stale_size_factor', 0.25))
                        if downgrade > 0 and downgrade < 1 and final_decision.vote_detail.get('risk_model'):
                            try:
                                rm = final_decision.vote_detail['risk_model']
                                rm['position_size'] = round(rm['position_size'] * downgrade, 2)
                                rm['profile_stale_scaled'] = True
                                rm['stale_factor'] = downgrade
                            except Exception:
                                pass
        except Exception:
            pass
        else:
            _trace_engine_flat(symbol, tf, ts_epoch, final_decision, "flat_after_combine")
            logger.debug("Final Decision for {} at {}: FLAT (Vote Detail: {})", symbol, timestamp, final_decision.vote_detail)

        # --- Sprint 17 Playbook layer (after ensemble + orderflow modulation, before sizing overlays) ---
        try:
            regime_obj = features.get("regime") if isinstance(features, dict) else None
            playbook = select_playbook(regime_obj, features, final_decision, self.settings)
            plan = None
            if playbook:
                plan = build_plan(playbook, {**features, "ohlcv": feature_vector.ohlcv}, final_decision, self.settings)
            if playbook and (not plan or getattr(playbook, 'abstain', False)):
                # convert to FLAT due to playbook abstain/gate
                final_decision.decision = "FLAT"
                final_decision.vote_detail.setdefault("playbook", {})
                final_decision.vote_detail["playbook"].update({
                    "reason": "PLAYBOOK_ABSTAIN",
                    "selected": getattr(playbook, 'name', None)
                })
                # top-level reason for consistency with earlier FLAT causes
                final_decision.vote_detail.setdefault("reason", "PLAYBOOK_ABSTAIN")
            elif plan:
                final_decision.vote_detail.setdefault("playbook", plan)
                # embed one-liner summary for downstream transport (telegram, etc.)
                summary = f"Playbook: {plan['reason']} | Stop:{plan['stop_atr_mult']}xATR | TPs:{plan['tp_atr_mults']} | RR>={round(plan['expected_rr'],2) if plan.get('expected_rr') else 'n/a'} | Size×{plan['size_scale']}"
                final_decision.vote_detail["playbook"]["summary"] = summary
                logger.debug(summary)
        except Exception as e:
            logger.exception("Playbook integration error: {}", e)

        # ------------------------------------------------------------------
        # Sprint 18 Quality Gates (Confidence Binning + Targeted Vetoes)
        # Execute AFTER playbook selection (need expected RR) and BEFORE sizing.
        # ------------------------------------------------------------------
        try:
            qcfg = (self.settings.get("quality_gates") or {}) if isinstance(self.settings, dict) else {}
            if final_decision.decision in ("LONG", "SHORT") and qcfg.get("enabled", True):
                # Merge plan into playbook dict for rr context (plan may contain expected_rr)
                pb_ctx = {}
                if isinstance(final_decision.vote_detail.get("playbook"), dict):
                    pb_ctx.update(final_decision.vote_detail.get("playbook"))
                qd = QualityGates.evaluate(features, final_decision, pb_ctx, self.settings)
                final_decision.vote_detail.setdefault("quality", {
                    "bin": qd.bin,
                    "qscore": qd.qscore,
                    "vetoes": qd.veto_reasons,
                    "soft_flags": qd.soft_flags
                })
                if qd.blocked:
                    final_decision.decision = "FLAT"
                    final_decision.vetoes.extend([f"VETO_{r}" for r in qd.veto_reasons])
                    final_decision.vote_detail.setdefault("reason", f"VETO:{','.join(qd.veto_reasons)}")
                else:
                    # mark required confirmations
                    if qd.requirements:
                        final_decision.vote_detail["quality"].update({"requirements": qd.requirements})
                    final_decision.vote_detail["quality"].update({"size_multiplier": qd.size_multiplier})
                    # store base size multiplier for later sizing overlay use
                    final_decision.vote_detail.setdefault("quality_size_mult", qd.size_multiplier)
        except Exception as e:
            logger.debug("Quality gates integration error: {}", e)

        # ------------------------------------------------------------------
        # Sprint 16: News & Volatility Veto System
        # Apply AFTER ensemble & orderflow modulation, BEFORE sizing layers.
        # ------------------------------------------------------------------
        try:
            # Lazy init / cache NewsVeto on engine instance
            nv_cfg = (self.settings.get("news_veto") or {}) if isinstance(self.settings, dict) else {}
            vol_veto_cfg = (self.settings.get("volatility_veto") or {}) if isinstance(self.settings, dict) else {}
            veto_applied = False
            if final_decision.decision in ("LONG", "SHORT"):
                # News embargo check
                if nv_cfg.get("enabled", True):
                    if not hasattr(self, "_news_veto"):
                        try:
                            self._news_veto = NewsVeto(self.settings)
                        except Exception:
                            self._news_veto = None
                    if getattr(self, "_news_veto", None):
                        blocked, reason = self._news_veto.is_event_now(symbol, ts_epoch * 1000)
                        if blocked:
                            final_decision.decision = "FLAT"
                            final_decision.vetoes.append("VETO_NEWS")
                            final_decision.vote_detail.setdefault("veto", {})
                            final_decision.vote_detail["veto"].update({
                                "news_event": reason,
                                "embargo_min": self._news_veto.embargo_minutes,
                            })
                            # Attach high-level reason & embed RiskEvent
                            final_decision.vote_detail.setdefault("reason", f"VETO_NEWS:{reason}")
                            try:
                                final_decision.vote_detail.setdefault("risk_events", []).append(RiskEvent(
                                    ts=ts_epoch,
                                    symbol=symbol,
                                    reason="NEWS_EMBARGO",
                                    action="VETO",
                                    detail={"event": reason}
                                ).__dict__)
                            except Exception:
                                pass
                            veto_applied = True

                # Volatility veto (ATR percentile, liquidation impulse, funding rate)
                if not veto_applied and vol_veto_cfg.get("enabled", True):
                    # Extract atr_percentile if present in regime or volatility features
                    atr_pct = None
                    try:
                        if features.get("regime") is not None:
                            atr_pct = getattr(features.get("regime"), "atr_percentile", None)
                    except Exception:
                        atr_pct = None
                    if atr_pct is None:
                        vol_obj = features.get("volatility")
                        if vol_obj is not None:
                            atr_pct = getattr(vol_obj, "atr_percentile", None)
                    # Normalize scale if user provided 0-100 vs 0-1
                    try:
                        if atr_pct and atr_pct > 1.0:
                            atr_pct = atr_pct / 100.0
                    except Exception:
                        pass
                    atr_limit = float(vol_veto_cfg.get("atr_percentile_limit", 0.95))
                    # Funding rate veto
                    funding_limit = float(vol_veto_cfg.get("funding_rate_limit", 0.0005))
                    funding_rate = None
                    try:
                        deriv = features.get("derivatives")
                        if deriv is not None:
                            funding_rate = getattr(deriv, "funding_now", None)
                    except Exception:
                        funding_rate = None
                    # Liquidation impulse from orderflow snapshot (already built)
                    liq_impulse = None
                    if of_snapshot is not None:
                        liq_impulse = of_snapshot.liq_impulse
                    # Sensitivity mapping: high -> lower impulse threshold
                    sens = str(vol_veto_cfg.get("liq_cluster_sensitivity", "high")).lower()
                    if sens == "high":
                        liq_impulse_th = 1.0
                    elif sens == "medium":
                        liq_impulse_th = 1.5
                    else:
                        liq_impulse_th = 2.0
                    reasons = {}
                    if atr_pct is not None and atr_pct >= atr_limit:
                        reasons["atr_percentile"] = atr_pct
                    if funding_rate is not None and abs(funding_rate) >= funding_limit:
                        reasons["funding_rate"] = funding_rate
                    if liq_impulse is not None and liq_impulse >= liq_impulse_th:
                        reasons["liq_impulse"] = liq_impulse
                    if reasons:
                        final_decision.decision = "FLAT"
                        # keep original code VETO_VOL for backward compatibility AND add specific spike code for ATR
                        if "atr_percentile" in reasons:
                            final_decision.vetoes.append("VETO_VOL_SPIKE")
                        else:
                            final_decision.vetoes.append("VETO_VOL")
                        final_decision.vote_detail.setdefault("veto", {})
                        final_decision.vote_detail["veto"].update(reasons)
                        # Provide concise reason at top level
                        if "atr_percentile" in reasons:
                            final_decision.vote_detail.setdefault("reason", "VETO_VOL_SPIKE")
                        elif "funding_rate" in reasons:
                            final_decision.vote_detail.setdefault("reason", "VETO_FUNDING")
                        elif "liq_impulse" in reasons:
                            final_decision.vote_detail.setdefault("reason", "VETO_LIQ_CLUSTER")
                        # RiskEvent embedding (for backtest summary extraction if desired)
                        try:
                            r_reason = "VOL_SPIKE" if "atr_percentile" in reasons else (
                                "FUNDING_EXTREME" if "funding_rate" in reasons else (
                                    "LIQ_CLUSTER" if "liq_impulse" in reasons else "VOL_VETO"
                                )
                            )
                            final_decision.vote_detail.setdefault("risk_events", []).append(RiskEvent(
                                ts=ts_epoch,
                                symbol=symbol,
                                reason=r_reason,
                                action="VETO",
                                detail=reasons
                            ).__dict__)
                        except Exception:
                            pass
                        veto_applied = True
        except Exception as e:
            logger.debug("Vol/News veto integration error: {}", e)

        # 7) Adaptive sizing & risk model (Sprint 12)
        try:
            if final_decision.decision in ("LONG", "SHORT"):
                # Provide features dict + regime object if available
                enriched_feats = dict(features)
                # attach regime obj under key for sizing logic
                if features.get("regime"):
                    enriched_feats["regime"] = features.get("regime")
                # Provide OHLCV slice to sizing module
                enriched_feats["ohlcv"] = feature_vector.ohlcv
                # Provide flow metrics if present
                if features.get("flow_metrics"):
                    enriched_feats["flow_metrics"] = features.get("flow_metrics")
                # Equity placeholder: could be integrated with portfolio module; for now read from settings or default
                equity = float((self.settings.get("portfolio", {}) or {}).get("mock_equity", 10_000.0))
                risk_decision = PositionSizing.calculate(symbol, final_decision, enriched_feats, self.settings, equity)
                if risk_decision:
                    final_decision.vote_detail.setdefault("risk_model", {
                        "position_size": round(risk_decision.size_quote, 2),
                        "leverage": risk_decision.leverage,
                        "stop_loss": round(risk_decision.stop_price, 4),
                        "take_profit": round(risk_decision.take_profit, 4),
                        "confidence": round(risk_decision.confidence, 4),
                        "regime": risk_decision.regime,
                        "atr": risk_decision.atr,
                        "risk_pct": risk_decision.reasoning.get("risk_pct"),
                    })
                    logger.debug("Risk Model Decision: {}", final_decision.vote_detail["risk_model"])

                    # Apply playbook size_scale (Sprint17) if plan present
                    try:
                        plan_pb = final_decision.vote_detail.get("playbook")
                        if isinstance(plan_pb, dict) and plan_pb.get("size_scale") and plan_pb.get("size_scale") != 1.0:
                            ps = final_decision.vote_detail["risk_model"].get("position_size")
                            scaled = ps * float(plan_pb["size_scale"])
                            final_decision.vote_detail["risk_model"]["position_size_playbook"] = round(scaled, 2)
                            final_decision.vote_detail["risk_model"]["size_scale_playbook"] = plan_pb["size_scale"]
                    except Exception:
                        pass

                # Apply quality size multiplier (Sprint 18) if present
                q_mult = float(final_decision.vote_detail.get("quality_size_mult", 1.0))
                if q_mult != 1.0 and final_decision.vote_detail.get("risk_model"):
                    try:
                        base_sz = final_decision.vote_detail["risk_model"]["position_size"]
                        final_decision.vote_detail["risk_model"]["position_size"] = round(base_sz * q_mult, 2)
                        final_decision.vote_detail["risk_model"]["quality_scaled"] = True
                        final_decision.vote_detail["risk_model"]["quality_size_mult"] = q_mult
                    except Exception:
                        pass

                # Sprint 15 Dynamic Position Sizing overlay (independent of adaptive model for now)
                ps_cfg = (self.settings.get("position_sizing") or {}) if isinstance(self.settings, dict) else {}
                if ps_cfg.get("enabled", True):
                    try:
                        # Liquidation heatmap risk
                        lh_cfg = (self.settings.get("liquidation_heatmap") or {})
                        heatmap = LiquidationHeatmap(self.settings)
                        clusters = heatmap.get_liq_levels(symbol) if lh_cfg.get("enabled", True) else []
                        current_price = float(feature_vector.ohlcv.get("close", 0.0)) if feature_vector.ohlcv else 0.0
                        liq_risk = LiquidationHeatmap.compute_liq_risk(current_price, clusters, heatmap.min_cluster_usd) if clusters else 0.0
                        # NEW (Sprint15 enhancement): compute nearest large cluster distance (basis points)
                        nearest_wall_bp = None
                        if clusters and current_price > 0:
                            import math  # lightweight
                            best_bp = None
                            for c in clusters:
                                try:
                                    if float(c.get('size', 0)) < heatmap.min_cluster_usd:
                                        continue
                                    dist_bp = abs(float(c.get('price')) - current_price) / current_price * 10_000
                                    if best_bp is None or dist_bp < best_bp:
                                        best_bp = dist_bp
                                except Exception:
                                    continue
                            if best_bp is not None:
                                nearest_wall_bp = round(best_bp, 2)
                        # Classify liq risk for reporting (low/medium/high)
                        def _classify_liq_risk(x: float) -> str:
                            if x is None:
                                return "unknown"
                            if x < 0.5:
                                return "low"
                            if x < 1.2:
                                return "medium"
                            return "high"
                        liq_risk_cls = _classify_liq_risk(liq_risk)
                        # compute atr (volatility) fallback
                        atr_val = 0.0
                        vol_obj = features.get("volatility")
                        if vol_obj is not None:
                            atr_val = float(getattr(vol_obj, "atr", 0.0) or 0.0)
                        conf = float(final_decision.confidence or 0.0)
                        if conf >= float(ps_cfg.get("min_confidence", 0.0)):
                            sizer = PositionSizer(
                                account_equity=equity,
                                max_risk_pct=float(ps_cfg.get("max_risk_pct", 0.02)),
                                atr_window=int(ps_cfg.get("atr_window", 14)),
                                liq_risk_weight=float(ps_cfg.get("liq_risk_weight", 0.6))
                            )
                            sz_res = sizer.calc_position_size(conf, atr_val, liq_risk)
                            # Optionally skip trade on extreme liq risk
                            if ps_cfg.get("skip_high_liq_risk", True) and liq_risk > float(ps_cfg.get("liq_risk_skip_threshold", 1.5)):
                                final_decision.decision = "FLAT"
                                final_decision.vote_detail.setdefault("position_sizer", {})
                                final_decision.vote_detail["position_sizer"].update({
                                    "skipped": True,
                                    "reason": "HIGH_LIQ_RISK",
                                    "liq_risk": liq_risk,
                    "liq_cluster_risk": liq_risk_cls,
                    "nearest_wall_bp": nearest_wall_bp,
                                    "clusters": clusters
                                })
                            else:
                                final_decision.vote_detail.setdefault("position_sizer", {})
                                final_decision.vote_detail["position_sizer"].update({
                                    "size_quote": round(sz_res.size_quote, 2),
                                    "base_risk": round(sz_res.base_risk, 2),
                                    "confidence": round(conf, 4),
                                    "atr": atr_val,
                                    "liq_risk": liq_risk,
                    "liq_cluster_risk": liq_risk_cls,
                    "nearest_wall_bp": nearest_wall_bp,
                                    "clusters": clusters,
                                    "clipped": sz_res.clipped
                                })
                                # Apply playbook size scaling if present
                                try:
                                    plan_pb = final_decision.vote_detail.get("playbook")
                                    if isinstance(plan_pb, dict) and plan_pb.get("size_scale") and plan_pb.get("size_scale") != 1.0:
                                        base_sz = final_decision.vote_detail["position_sizer"]["size_quote"]
                                        scaled = base_sz * float(plan_pb["size_scale"])
                                        final_decision.vote_detail["position_sizer"]["size_quote_playbook"] = round(scaled, 2)
                                        final_decision.vote_detail["position_sizer"]["size_scale_playbook"] = plan_pb["size_scale"]
                                except Exception:
                                    pass
                    except Exception:
                        pass
        except Exception as e:
            logger.exception("Adaptive sizing error: {}", e)

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
