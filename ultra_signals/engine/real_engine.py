from typing import Dict, Optional, Any, List

import pandas as pd
from loguru import logger
from time import perf_counter  # Sprint 30 performance timing for MTC

from ultra_signals.core.feature_store import FeatureStore
from ultra_signals.core.custom_types import EnsembleDecision, FeatureVector, SubSignal, RiskEvent
from ultra_signals.engine import ensemble, regime, scoring
from ultra_signals.risk.position_sizing import PositionSizing
from ultra_signals.engine.regime_router import RegimeRouter
from ultra_signals.engine.orderflow import OrderFlowAnalyzer, OrderFlowSnapshot, apply_orderflow_modulation
try:
    # optional lightweight orderflow engine (new S51 module)
    from ultra_signals.orderflow.engine import OrderflowEngine
except Exception:
    OrderflowEngine = None
from ultra_signals.engine.liquidation_heatmap import LiquidationHeatmap
from ultra_signals.engine.position_sizer import PositionSizer
from ultra_signals.engine.sizing.advanced_sizer import AdvancedSizer
from ultra_signals.engine.execution_planner import select_playbook, build_plan
from ultra_signals.engine.news_veto import NewsVeto
from ultra_signals.engine.quality_gates import QualityGates
from ultra_signals import events  # Sprint 28 events gating
from ultra_signals.engine.gates import liquidity_gate, LiquidityGate  # Sprint 29 liquidity gate
from ultra_signals.engine.gates import evaluate_mtc_gate  # Sprint 30 MTC gate
from ultra_signals.engine.gates import evaluate_meta_gate  # Sprint 31 Meta probability gate
from ultra_signals.engine.gates.whale_gate import evaluate_whale_gate  # Sprint 41 Whale gate
from ultra_signals.behavior import BehaviorEngine  # Sprint 45 behavioral veto
from ultra_signals.features.htf_cache import HTFFeatureCache
# Sprint 22 optional portfolio hedging modules (guarded)
try:  # pragma: no cover
    from ultra_signals.portfolio.correlations import RollingCorrelationBeta
    from ultra_signals.portfolio.exposure import PortfolioExposure
    from ultra_signals.portfolio.hedger import BetaHedger
    from ultra_signals.portfolio.risk_caps import PortfolioRiskCaps
    from ultra_signals.portfolio.hedge_report import HedgeReportCollector, HedgeSnapshot
    # Sprint 33 new modules
    from ultra_signals.portfolio.risk_estimator import RiskEstimator
    from ultra_signals.portfolio.allocator import PortfolioAllocator
except Exception:  # If missing, engine continues without hedging
    RollingCorrelationBeta = PortfolioExposure = BetaHedger = PortfolioRiskCaps = HedgeReportCollector = HedgeSnapshot = None
    RiskEstimator = PortfolioAllocator = None


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
        # Sprint 32 advanced sizer
        try:
            self._adv_sizer = AdvancedSizer(self.settings)
        except Exception:
            self._adv_sizer = None
        # Sprint 22: optional portfolio hedge state
        ph_cfg = (self.settings.get("portfolio_hedge") or {}) if isinstance(self.settings, dict) else {}
        self._ph_enabled = bool(ph_cfg.get("enabled", False)) and RollingCorrelationBeta is not None
        if self._ph_enabled:
            try:
                self._ph_corr = RollingCorrelationBeta(
                    leader=ph_cfg.get("leader", "BTCUSDT"),
                    lookback=int(ph_cfg.get("lookback_bars", 288)),
                    shrinkage_lambda=float(ph_cfg.get("shrinkage_lambda", 0.0)),
                )
                self._ph_corr.corr_threshold_high = float(ph_cfg.get("corr_threshold_high", 0.55))
                self._ph_exposure = PortfolioExposure(cluster_map=ph_cfg.get("cluster_map", {}))
                band = ph_cfg.get("beta_band", {"min": -0.15, "max": 0.15})
                self._ph_caps = PortfolioRiskCaps(
                    beta_band=(float(band.get("min", -0.15)), float(band.get("max", 0.15))),
                    beta_hard_cap=float(ph_cfg.get("beta_hard_cap", 0.25)),
                    block_if_exceeds_beta_cap=bool((ph_cfg.get("open_guard", {}) or {}).get("block_if_exceeds_beta_cap", True)),
                    downscale_if_over_band=bool((ph_cfg.get("open_guard", {}) or {}).get("downscale_if_over_band", True)),
                    downscale_factor=float((ph_cfg.get("open_guard", {}) or {}).get("downscale_factor", 0.5)),
                    cluster_caps=ph_cfg.get("cluster_caps", {}),
                )
                self._ph_hedger = BetaHedger(
                    leader=ph_cfg.get("leader", "BTCUSDT"),
                    beta_band=(float(band.get("min", -0.15)), float(band.get("max", 0.15))),
                    min_rebalance_frac=float((ph_cfg.get("rebalance", {}) or {}).get("min_notional", 0.005)),
                    taker_fee=float((ph_cfg.get("costs", {}) or {}).get("taker_fee", 0.0004)),
                    cooloff_bars=int((ph_cfg.get("rebalance", {}) or {}).get("cooloff_bars", 3)),
                )
                self._ph_report = HedgeReportCollector()
                self._ph_equity = float((self.settings.get("portfolio", {}) or {}).get("mock_equity", 10_000.0))
                # Leader bias configuration (optional)
                self._ph_bias_cfg = (ph_cfg.get("leader_bias") or {}) if isinstance(ph_cfg, dict) else {}
            except Exception as e:  # pragma: no cover
                logger.warning("Portfolio hedge init failed: {}", e)
                self._ph_enabled = False
        # Sprint 29: persistent liquidity gate (cooldown stateful)
        try:
            self._lq_gate = LiquidityGate(self.settings)
        except Exception:
            self._lq_gate = None
        # Sprint 30 HTF cache for MTC
        try:
            self._htf_cache = HTFFeatureCache(feature_store, self.settings)
        except Exception:
            self._htf_cache = None
        # Sprint 45: Behavioral finance engine
        try:
            self._behavior_eng = BehaviorEngine(self.settings, feature_store)
        except Exception:
            self._behavior_eng = None
        # Sprint 30: perf accumulator for MTC gate (initialize after caches)
        try:
            self._mtc_perf = {"count": 0, "total_ms": 0.0, "max_ms": 0.0}
        except Exception:
            self._mtc_perf = {"count": 0, "total_ms": 0.0, "max_ms": 0.0}
        # Sprint 33: external open positions snapshot (injected by runner / live execution layer)
        self._external_open_positions = []  # list[dict]
        # Sprint 33: timeseries metrics accumulation for reporting
        self._pr_metrics_ts = []  # list[dict]
        # Sprint 46: Economic Event Service injection (lazy; user may disable)
        try:
            econ_cfg = (self.settings.get('econ') or {}) if isinstance(self.settings, dict) else {}
            if econ_cfg.get('enabled', False):
                from ultra_signals.econ.service import EconEventService, static_config_collector_factory
                self._econ_service = EconEventService(econ_cfg)
                # Optional static events defined in config for tests / offline
                static_events = econ_cfg.get('static_events') or []
                if static_events:
                    self._econ_service.register_collector('static', static_config_collector_factory(static_events))
            else:
                self._econ_service = None
        except Exception as e:  # pragma: no cover
            logger.debug('Econ service init failed: {}', e)
            self._econ_service = None
        # Orderflow engine (S51) - optional lightweight in-process engine for micro features
        try:
            if OrderflowEngine is not None:
                self._of_engine = OrderflowEngine((self.settings.get('orderflow') or {}) if isinstance(self.settings, dict) else {})
            else:
                self._of_engine = None
        except Exception:
            self._of_engine = None
        # Drift policy engine (optional): lazy init to avoid adding hard deps
        try:
            from ultra_signals.drift.policy import PolicyEngine
            self._policy_engine = PolicyEngine((self.settings.get('drift_policy') or {}) if isinstance(self.settings, dict) else {})
        except Exception:
            self._policy_engine = None
        # --- L-VaR, ExecAdapter, CircuitBreaker integration (Sprint 53) ---
        try:
            from ultra_signals.risk.lvar import LVarEngine
            from ultra_signals.risk.exec_adapter import ExecAdapter
            from ultra_signals.risk.circuit_breaker import CircuitBreaker
            risk_cfg = (self.settings.get('risk') or {}) if isinstance(self.settings, dict) else {}
            equity_override = float((self.settings.get('portfolio', {}) or {}).get('mock_equity', 10000.0))
            self._lvar_engine = LVarEngine(equity=equity_override, pr_cap=float(risk_cfg.get('pr_cap', 0.12)))
            self._exec_adapter = ExecAdapter(liq_cost_max_pct_equity=float(risk_cfg.get('liq_cost_max_pct_equity', 0.01)), lvar_max_pct_equity=float(risk_cfg.get('lvar_max_pct_equity', 0.02)))
            self._circuit_breaker = CircuitBreaker(k_sigma=float(risk_cfg.get('flash_k_sigma', 6.0)), cooldown_bars=int(risk_cfg.get('cooldown_bars', 5)))
        except Exception:
            # keep engine resilient if risk modules missing
            self._lvar_engine = None
            self._exec_adapter = None
            self._circuit_breaker = None

    # ----------------- Sprint 33 helpers (open positions + metrics) -----------------
    def set_open_positions(self, positions: list[dict]) -> None:
        """Inject current live/backtest open positions.
        Expected schema per position (keys used if present):
          symbol, side (LONG/SHORT), qty, entry_price, stop_price, risk_amount, cluster
        risk_amount optional: if missing will attempt to derive later.
        """
        try:
            if isinstance(positions, list):
                self._external_open_positions = positions
        except Exception:
            pass

    def get_portfolio_risk_metrics_ts(self) -> list[dict]:
        """Return accumulated per-bar portfolio risk metrics timeseries (shallow copy)."""
        try:
            return list(self._pr_metrics_ts)
        except Exception:
            return []
    # -------------------------------------------------------------------------------

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
            regime=features.get("regime"),
            derivatives=None,
            orderbook=None,
            rs=None,
            flow_metrics=features.get("flow_metrics")
        )
        # Sprint 46 econ feature attachment (built early so downstream gates can observe)
        try:
            if getattr(self, '_econ_service', None):
                # Refresh service if cadence due, then build current features
                self._econ_service.refresh(int(ts_epoch*1000))
                econ_feats = self._econ_service.build_features(int(ts_epoch*1000))
                feature_vector.econ = econ_feats
        except Exception as e:  # pragma: no cover
            logger.debug('Econ feature build error: {}', e)
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
                # If we have the in-process OrderflowEngine, feed recent trades/ob for richer metrics
                try:
                    if getattr(self, '_of_engine', None) is not None:
                        now_ts = int(ts_epoch or int(time.time()))
                        # ingest trades
                        for t in trades:
                            t_ts = int(t.get('ts') or t.get('timestamp') or now_ts)
                            price = float(t.get('price') or t.get('px') or 0.0)
                            qty = float(t.get('qty') or t.get('quantity') or 0.0)
                            side = t.get('side')
                            if side is None and 'is_buyer_maker' in t:
                                side = 'sell' if t['is_buyer_maker'] else 'buy'
                            self._of_engine.ingest_trade(t_ts, price, qty, side, aggressor=True)
                        # ingest orderbook top levels if available
                        if ob:
                            bids = ob.get('bids') or []
                            asks = ob.get('asks') or []
                            self._of_engine.ingest_orderbook_snapshot(bids, asks)
                except Exception:
                    pass
                of_snapshot = OrderFlowAnalyzer.build_snapshot(trades, liqs, ob, self.settings, prev_cvd)
            except Exception:
                of_snapshot = None

        # Sprint 22: update rolling correlations (cheap incremental append)
        if self._ph_enabled:
            try:
                close_px = float(feature_vector.ohlcv.get("close", 0.0)) if feature_vector and feature_vector.ohlcv else None
                if close_px and close_px > 0:
                    self._ph_corr.update_price(symbol, close_px, ts_epoch)
                    # Recompute betas when leader updates or every 25 leader bars equivalently for others
                    if symbol == self._ph_corr.leader or (len(self._ph_corr._prices.get(symbol, [])) % 25 == 0):
                        self._ph_corr.recompute()
            except Exception:
                pass

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

        # --- Sprint 53: Flash-crash Circuit Breaker check (early) ---
        try:
            if getattr(self, '_circuit_breaker', None) is not None:
                # use last return and vol forecast if present
                ret_pct = None
                sigma = None
                try:
                    # compute last bar return
                    prev_close = None
                    df = self.feature_store.get_ohlcv(symbol, tf)
                    if df is not None and len(df) >= 2:
                        prev_close = float(df['close'].iloc[-2])
                        last_close = float(df['close'].iloc[-1])
                        if prev_close and last_close:
                            ret_pct = (last_close - prev_close) / prev_close
                except Exception:
                    ret_pct = None
                try:
                    vol_obj = features.get('volatility') if isinstance(features, dict) else None
                    sigma = float(getattr(vol_obj, 'sigma', None) or getattr(vol_obj, 'atr', None) or 0.0)
                except Exception:
                    sigma = None
                cb_state = self._circuit_breaker.check_and_trigger(ret_pct if ret_pct is not None else 0.0, sigma if sigma is not None else 0.0, spread_z=(features.get('impact') or {}).get('impact_hints', None) and getattr((features.get('impact') or {}).get('impact_hints'), 'impact_state', None), vpin_toxic=False)
                if cb_state and cb_state.triggered:
                    final_decision.decision = 'FLAT'
                    final_decision.vetoes.append('FLASH_CB')
                    final_decision.vote_detail.setdefault('circuit_breaker', cb_state.__dict__)
                    _trace_engine_flat(symbol, tf, ts_epoch, final_decision, 'flash_circuit_breaker')
                    return final_decision
        except Exception:
            pass

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
        # Sprint 45: Behavioral Engine evaluation (early attach so downstream gates can see)
        try:
            if getattr(self, '_behavior_eng', None):
                beh_bundle = {
                    'ohlcv': feature_vector.ohlcv,
                    'flow_metrics': features.get('flow_metrics'),
                    'derivatives': features.get('derivatives'),
                    'regime': features.get('regime'),
                    'sentiment': final_decision.vote_detail.get('sentiment') if isinstance(final_decision.vote_detail, dict) else None,
                    'whales': final_decision.vote_detail.get('whales') if isinstance(final_decision.vote_detail, dict) else None,
                }
                beh_feats = self._behavior_eng.evaluate(symbol, ts_epoch, beh_bundle)
                if beh_feats:
                    # attach into feature_vector for optional logging & downstream
                    try:
                        feature_vector.behavior = beh_feats
                    except Exception:
                        pass
                    final_decision.vote_detail.setdefault('behavior', beh_feats.model_dump() if hasattr(beh_feats,'model_dump') else beh_feats.__dict__)
                    # Apply behavioral action pre-quality-gates: if veto -> immediate FLAT with code
                    if final_decision.decision in ('LONG','SHORT') and beh_feats.behavior_action == 'VETO':
                        final_decision.decision = 'FLAT'
                        final_decision.vetoes.append('BEHAVIOR_VETO')
                        final_decision.vote_detail.setdefault('reason','BEHAVIOR_VETO')
                    elif final_decision.decision in ('LONG','SHORT') and beh_feats.behavior_action == 'DAMPEN':
                        # store multiplier to apply after base risk model sizing (layered)
                        final_decision.vote_detail.setdefault('behavior_size_mult', beh_feats.behavior_size_mult or 1.0)
        except Exception as e:  # pragma: no cover
            logger.debug('Behavior engine error: {}', e)
        # Sprint 42: attach latest macro features (if computed) for transport & macro gating transparency
        try:
            macro_feats = features.get('macro') if isinstance(features, dict) else None
            if macro_feats:
                snap = macro_feats.model_dump() if hasattr(macro_feats, 'model_dump') else dict(macro_feats)
                final_decision.vote_detail.setdefault('macro', snap)
                # expose snapshot to ensemble (used for gating) via transient settings key
                # (safe: ephemeral; not persisted)
                self.settings['_latest_macro'] = snap
        except Exception:
            pass
        # Attach simple regime snapshot for downstream transports / telemetry
        try:
            reg_obj = features.get("regime")
            if reg_obj is not None:
                final_decision.vote_detail.setdefault("regime", {
                    "primary": getattr(reg_obj, "profile", None).value if getattr(reg_obj, "profile", None) else None,
                    "vol": getattr(reg_obj, "vol_state", None).value if getattr(reg_obj, "vol_state", None) else None,
                    "liq": getattr(reg_obj, "liquidity", None).value if getattr(reg_obj, "liquidity", None) else None,
                    "confidence": round(float(getattr(reg_obj, "confidence", 0.0) or 0.0), 3),
                })
        except Exception:
            pass
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
        # Sprint 30 MTC Gate (after ensemble+orderflow modulation, before quality/news/liquidity)
        # ------------------------------------------------------------------
        mtc_outcome = None
        try:
            if final_decision.decision in ("LONG", "SHORT"):
                mtc_cfg = (self.settings.get("mtc") or {}) if isinstance(self.settings, dict) else {}
                if mtc_cfg.get("enabled", True) and getattr(self, "_htf_cache", None):
                    ladders = (mtc_cfg.get("ladders") or {})
                    ladder = ladders.get(tf) or ladders.get(tf.lower())
                    htf_map = {}
                    if ladder:
                        if len(ladder) >= 1:
                            htf_map["C1"] = self._htf_cache.get_htf_features(symbol, ladder[0], ts_epoch)
                        if len(ladder) >= 2:
                            htf_map["C2"] = self._htf_cache.get_htf_features(symbol, ladder[1], ts_epoch)
                    # Drop None values
                    htf_map = {k: v for k, v in htf_map.items() if v is not None}
                    t0 = perf_counter()
                    mtc_outcome = evaluate_mtc_gate(final_decision.decision, symbol, tf, ts_epoch, current_regime, self.settings, htf_map)
                    elapsed_ms = (perf_counter() - t0) * 1000.0
                    # Accumulate perf stats
                    try:
                        self._mtc_perf["count"] += 1
                        self._mtc_perf["total_ms"] += elapsed_ms
                        if elapsed_ms > self._mtc_perf["max_ms"]:
                            self._mtc_perf["max_ms"] = elapsed_ms
                        # Periodic log every 200 evals
                        if self._mtc_perf["count"] % 200 == 0:
                            avg = self._mtc_perf["total_ms"] / max(1, self._mtc_perf["count"])
                            logger.debug(f"[MTC] perf avg={avg:.3f}ms max={self._mtc_perf['max_ms']:.3f}ms n={self._mtc_perf['count']}")
                    except Exception:
                        pass
                    observe_only = bool(mtc_cfg.get("observe_only", False))
                    final_decision.vote_detail.setdefault("mtc_gate", {
                        "action": getattr(mtc_outcome, 'action', None),
                        "status": getattr(mtc_outcome, 'status', None),
                        "scores": getattr(mtc_outcome, 'scores', {}),
                        "reasons": getattr(mtc_outcome, 'reasons', []),
                        "elapsed_ms": round(elapsed_ms, 3),
                        "observe_only": observe_only,
                    })
                    # Enforce only if not observe-only
                    if not observe_only:
                        if mtc_outcome.action == "VETO":
                            final_decision.vetoes.append("MTC_VETO")
                            final_decision.decision = "FLAT"
                            final_decision.vote_detail.setdefault("reason", "MTC_FAIL")
                    else:
                        # Normalize action semantics to ENTER so downstream logic doesn't dampen/ veto
                        # but preserve original under meta_original_action
                        try:
                            md = final_decision.vote_detail.get("mtc_gate")
                            if md:
                                md["original_action"] = mtc_outcome.action
                                md["action"] = "ENTER"
                        except Exception:
                            pass
        except Exception as e:  # pragma: no cover
            logger.debug("MTC gate error: {}", e)
        # ------------------------------------------------------------------
        # Sprint 31 Meta Probability Gate (after MTC, before quality gates)
        # ------------------------------------------------------------------
        try:
            meta_cfg = (self.settings.get('meta_scorer') or {}) if isinstance(self.settings, dict) else {}
            if final_decision.decision in ("LONG","SHORT") and meta_cfg.get('enabled', True):
                # Build minimal feature bundle from existing feature dict (flatten simple numeric attrs)
                bundle = {}
                try:
                    for k,v in (features or {}).items():
                        if v is None: continue
                        # If object has a dict-like or simple attributes, extract whitelisted numeric fields
                        if isinstance(v, (int,float)):
                            bundle[k]=v
                        else:
                            for attr in ['ema21','ema200','adx','rsi','macd_line','macd_signal','macd_hist','macd_slope','atr_percentile','bb_width','vwap','price','spread_bps','impact_50k','dr','rv_5s','volume_z']:
                                if hasattr(v, attr):
                                    val = getattr(v, attr)
                                    if isinstance(val,(int,float)):
                                        bundle[attr]=val
                except Exception:
                    pass
                # Regime profile
                regime_profile = 'trend'
                try:
                    reg = features.get('regime') if isinstance(features, dict) else None
                    if reg and getattr(reg,'profile',None):
                        regime_profile = str(getattr(reg,'profile').value)
                except Exception:
                    pass
                # Attach orderflow micro score if available
                try:
                    if getattr(self, '_of_engine', None) is not None:
                        m = self._of_engine.compute_micro_score()
                        # expose scalar and components
                        bundle['of_micro_score'] = float(m.get('of_micro_score') or 0.0)
                        for ck, cv in (m.get('components') or {}).items():
                            bundle[f'of_comp_{ck}'] = float(cv or 0.0)
                except Exception:
                    pass
                t0 = perf_counter()
                meta_decision = evaluate_meta_gate(final_decision.decision, regime_profile, bundle, self.settings)
                elapsed_ms = (perf_counter()-t0)*1000.0
                final_decision.vote_detail.setdefault('meta_gate', {
                    'p': meta_decision.p,
                    'action': meta_decision.action,
                    'reason': meta_decision.reason,
                    'threshold': meta_decision.threshold,
                    'profile': meta_decision.profile,
                    'elapsed_ms': round(elapsed_ms,3),
                    'band': meta_decision.meta,
                })
                observe_only = bool(meta_cfg.get('shadow_mode', False))
                final_decision.vote_detail['meta_gate']['shadow_mode'] = observe_only
                if not observe_only:
                    if meta_decision.action == 'VETO':
                        final_decision.decision = 'FLAT'
                        final_decision.vetoes.append('META_VETO')
                        final_decision.vote_detail.setdefault('reason','META_LOW_PROB')
                    elif meta_decision.action == 'DAMPEN':
                        # apply size / stop adjustments if risk model computed later
                        final_decision.vote_detail['meta_gate']['size_mult'] = meta_decision.size_mult
                        final_decision.vote_detail['meta_gate']['widen_stop_mult'] = meta_decision.widen_stop_mult
                else:
                    # Normalize action in shadow mode for downstream compatibility
                    final_decision.vote_detail['meta_gate']['original_action'] = meta_decision.action
                    final_decision.vote_detail['meta_gate']['action'] = 'ENTER'
        except Exception as e:  # pragma: no cover
            logger.debug('Meta gate error: {}', e)
        # Sprint 43 — Regime snapshot injection for downstream transports (Telegram, logs)
        try:
            reg_obj = feature_vector.regime if hasattr(feature_vector, 'regime') else None
            if reg_obj:
                final_decision.vote_detail.setdefault('regime', {})
                vd = final_decision.vote_detail['regime']
                for attr in [
                    'regime_label','regime_probs','transition_hazard','exp_vol_h','dir_bias',
                    'macro_risk_context','sent_extreme_flag','whale_pressure'
                ]:
                    val = getattr(reg_obj, attr, None)
                    if val is not None:
                        vd[attr] = val
        except Exception:
            pass
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
        # Sprint 46 Economic Event Gate (after quality gates, before legacy news/vol). Applies
        # veto or size dampen based on econ feature pack produced earlier.
        # ------------------------------------------------------------------
        try:
            if final_decision.decision in ('LONG','SHORT') and getattr(feature_vector, 'econ', None):
                econ_feats = feature_vector.econ
                econ_cfg = (self.settings.get('econ') or {}) if isinstance(self.settings, dict) else {}
                # Veto if policy says severity high and inside window and allow_veto enabled
                sev = getattr(econ_feats, 'econ_severity', None)
                side = getattr(econ_feats, 'econ_window_side', None)
                size_mult = getattr(econ_feats, 'allowed_size_mult_econ', None)
                allow_veto = bool(econ_cfg.get('apply_veto', True))
                if sev == 'high' and side in ('pre','live') and allow_veto and econ_feats.econ_risk_active:
                    final_decision.decision = 'FLAT'
                    final_decision.vetoes.append('ECON_VETO')
                    final_decision.vote_detail.setdefault('reason','ECON_VETO')
                    final_decision.vote_detail.setdefault('econ_gate', {'action':'VETO','severity':sev,'side':side})
                elif size_mult is not None and size_mult < 1.0:
                    final_decision.vote_detail.setdefault('econ_gate', {'action':'DAMPEN','severity':sev,'side':side,'size_mult':size_mult})
        except Exception as e:  # pragma: no cover
            logger.debug('Econ gate error: {}', e)

        # ------------------------------------------------------------------
        # Sprint 28 Event Gate (macro / crypto incidents)
        # Executed AFTER quality gates but BEFORE legacy news/vol veto system so
        # newer event_risk can supersede older simplistic news_veto.
        # ------------------------------------------------------------------
        try:
            if final_decision.decision in ("LONG","SHORT"):
                pre = final_decision.decision  # capture before gate
                gate = events.evaluate_gate(symbol, ts_epoch*1000, None, None, self.settings)
                if gate.action == 'VETO':
                    final_decision.vote_detail.setdefault('event_gate', gate.__dict__)
                    final_decision.vote_detail['event_gate']['pre_decision'] = pre
                    final_decision.vote_detail['event_gate']['force_close'] = gate.force_close
                    final_decision.decision = 'FLAT'
                    final_decision.vetoes.append(f"NEWS_VETO:{gate.category or gate.reason}")
                    final_decision.vote_detail.setdefault('reason', f"NEWS_VETO:{gate.category or gate.reason}")
                elif gate.action == 'DAMPEN':
                    final_decision.vote_detail.setdefault('event_gate', gate.__dict__)
                    final_decision.vote_detail['event_gate']['pre_decision'] = pre
                    # widen stop if risk_model already present
                    try:
                        if gate.widen_stop_mult and final_decision.vote_detail.get('risk_model'):
                            rm = final_decision.vote_detail['risk_model']
                            if rm.get('stop_loss') and rm.get('atr'):
                                rm['stop_loss'] = rm['stop_loss'] * float(gate.widen_stop_mult)
                                rm['event_gate_stop_widened'] = gate.widen_stop_mult
                    except Exception:
                        pass
        except Exception as e:  # pragma: no cover
            logger.debug("Event gate integration error: {}", e)

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

        # ------------------------------------------------------------------
        # Sprint 29 Liquidity / Microstructure Gate
        # Executed AFTER news/vol veto but BEFORE sizing so size / stops can
        # be modified. We evaluate gate -> if VETO convert to FLAT + mark
        # veto code; if DAMPEN store modifiers to apply post sizing.
        # ------------------------------------------------------------------
        lq_gate_outcome = None
        try:
            if final_decision.decision in ("LONG", "SHORT"):
                # Determine regime profile mapping (trend/mean_revert/chop)
                profile = "trend"
                try:
                    reg_obj = features.get("regime") if isinstance(features, dict) else None
                    if reg_obj and getattr(reg_obj, "profile", None):
                        profile = str(getattr(reg_obj, "profile").value)
                except Exception:
                    pass
                # Acquire BookHealth snapshot if FeatureStore exposes it; else None (gate will apply missing policy)
                book = None
                try:
                    get_bh = getattr(self.feature_store, "get_latest_book_health", None)
                    if callable(get_bh):
                        book = get_bh(symbol)
                except Exception:
                    book = None
                # attach of_micro_score into book meta if engine available
                try:
                    if getattr(self, '_of_engine', None) is not None and book is not None:
                        m = self._of_engine.compute_micro_score()
                        # best-effort attach
                        try:
                            setattr(book, 'of_micro_score', float(m.get('of_micro_score') or 0.0))
                            setattr(book, 'of_micro_components', dict(m.get('components') or {}))
                        except Exception:
                            # fallback: store inside meta dict if exists
                            try:
                                if hasattr(book, 'meta') and isinstance(book.meta, dict):
                                    book.meta['of_micro_score'] = float(m.get('of_micro_score') or 0.0)
                            except Exception:
                                pass
                except Exception:
                    pass
                lq_gate_outcome = liquidity_gate.evaluate_gate(symbol, ts_epoch, profile, book, self.settings, getattr(self, "_lq_gate", None))
                # Persist decision (best-effort)
                try:
                    if lq_gate_outcome:
                        from ultra_signals.persist.db import record_liquidity_decision  # local import to avoid circulars
                        meta = dict(lq_gate_outcome.meta or {})
                        # augment meta with raw metrics if BookHealth present
                        if book is not None:
                            try:
                                meta.setdefault('spread_bps', getattr(book, 'spread_bps', None))
                                meta.setdefault('impact_50k', getattr(book, 'impact_50k', None))
                                meta.setdefault('dr', getattr(book, 'dr', None))
                                meta.setdefault('rv_5s', getattr(book, 'rv_5s', None))
                                meta.setdefault('source', getattr(book, 'source', None))
                            except Exception:
                                pass
                        record_liquidity_decision(symbol, ts_epoch, profile, lq_gate_outcome.action, lq_gate_outcome.reason, meta)
                except Exception:
                    pass
                if lq_gate_outcome.action == "VETO":
                    final_decision.decision = "FLAT"
                    code = f"LQ_VETO:{lq_gate_outcome.reason}" if lq_gate_outcome.reason else "LQ_VETO"
                    final_decision.vetoes.append(code)
                    final_decision.vote_detail.setdefault("liquidity_gate", lq_gate_outcome.__dict__)
                    final_decision.vote_detail.setdefault("reason", code)
                elif lq_gate_outcome.action == "DAMPEN":
                    final_decision.vote_detail.setdefault("liquidity_gate", lq_gate_outcome.__dict__)
                else:
                    # still record a NONE outcome for telemetry consistency
                    final_decision.vote_detail.setdefault("liquidity_gate", lq_gate_outcome.__dict__)
        except Exception as e:  # pragma: no cover
            logger.debug("Liquidity gate error: {}", e)

        # ------------------------------------------------------------------
        # Sprint 41 Whale / Smart Money Gate (after liquidity, before sizing)
        # ------------------------------------------------------------------
        whale_gate_outcome = None
        try:
            if final_decision.decision in ("LONG", "SHORT"):
                wf_obj = features.get('whales') if isinstance(features, dict) else None
                if wf_obj is not None:
                    whale_gate_outcome = evaluate_whale_gate(wf_obj, self.settings)
                    # Record outcome
                    final_decision.vote_detail.setdefault('whale_gate', {
                        'action': getattr(whale_gate_outcome, 'action', None),
                        'reason': getattr(whale_gate_outcome, 'reason', None),
                        'size_mult': getattr(whale_gate_outcome, 'size_mult', None),
                        'meta': getattr(whale_gate_outcome, 'meta', None),
                    })
                    if whale_gate_outcome.action == 'VETO':
                        final_decision.decision = 'FLAT'
                        code = f"WHALE_VETO:{whale_gate_outcome.reason}" if whale_gate_outcome.reason else 'WHALE_VETO'
                        final_decision.vetoes.append(code)
                        final_decision.vote_detail.setdefault('reason', code)
        except Exception as e:  # pragma: no cover
            logger.debug('Whale gate error: {}', e)

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
                    # Sprint 34 adaptive exits overlay (non-destructive)
                    try:
                        from ultra_signals.engine.adaptive_exits import generate_adaptive_exits
                        # Build minimal ohlcv tail: attempt to pull from feature_store (last atr_lookback*2 bars)
                        ae_cfg = ((self.settings.get('risk') or {}).get('adaptive_exits') or {})
                        lookback = int(ae_cfg.get('atr_lookback', 14)) * 3
                        ohlcv_tail = None
                        try:
                            get_hist = getattr(self.feature_store, 'get_ohlcv_history', None)
                            if callable(get_hist):
                                ohlcv_tail = get_hist(symbol, tf, lookback)
                        except Exception:
                            ohlcv_tail = None
                        regime_info = {}
                        try:
                            reg_obj = features.get('regime') if isinstance(features, dict) else None
                            if reg_obj is not None:
                                regime_info = {
                                    'profile': getattr(reg_obj, 'profile', None).value if getattr(reg_obj, 'profile', None) else getattr(reg_obj,'profile', None),
                                    'vol_state': getattr(reg_obj, 'vol_state', None).value if getattr(reg_obj, 'vol_state', None) else getattr(reg_obj,'vol_state', None),
                                    'tf': tf,
                                }
                        except Exception:
                            regime_info = {'tf': tf}
                        exits = generate_adaptive_exits(symbol, final_decision.decision, float(feature_vector.ohlcv.get('close',0.0)), ohlcv_tail, regime_info, self.settings, atr_current=risk_decision.atr)
                        if exits:
                            final_decision.vote_detail.setdefault('adaptive_exits', exits)
                            # Optionally override base stop/tp only if config enabled & more conservative (stop tighter) or more expansive target
                            try:
                                rm = final_decision.vote_detail.get('risk_model')
                                if rm and exits.get('stop_price') and exits.get('target_price'):
                                    # Replace if different; keep original under shadow keys
                                    rm.setdefault('orig_stop_loss', rm.get('stop_loss'))
                                    rm.setdefault('orig_take_profit', rm.get('take_profit'))
                                    rm['stop_loss'] = exits['stop_price']
                                    rm['take_profit'] = exits['target_price']
                                    rm['adaptive_applied'] = True
                            except Exception:
                                pass
                    except Exception:
                        pass

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

                # Sprint 45: Apply behavioral size multiplier AFTER quality scaling but BEFORE meta/mtc/liquidity layering
                try:
                    b_mult = float(final_decision.vote_detail.get('behavior_size_mult', 1.0))
                    if b_mult != 1.0 and final_decision.vote_detail.get('risk_model'):
                        base_sz = final_decision.vote_detail['risk_model']['position_size']
                        final_decision.vote_detail['risk_model']['position_size'] = round(base_sz * b_mult,2)
                        final_decision.vote_detail['risk_model']['behavior_scaled'] = True
                        final_decision.vote_detail['risk_model']['behavior_size_mult'] = b_mult
                except Exception:
                    pass

                # Sprint 31: Meta gate partial dampen (apply AFTER quality scaling, BEFORE MTC)
                try:
                    mg = final_decision.vote_detail.get("meta_gate")
                    if mg and mg.get("action") == "DAMPEN" and final_decision.vote_detail.get("risk_model"):
                        size_mult = float(mg.get("size_mult") or 1.0)
                        widen_mult = float(mg.get("widen_stop_mult") or 1.0)
                        if 0 < size_mult < 1.0:
                            base_sz = final_decision.vote_detail["risk_model"]["position_size"]
                            final_decision.vote_detail["risk_model"]["position_size"] = round(base_sz * size_mult, 2)
                            final_decision.vote_detail["risk_model"]["meta_scaled"] = True
                            final_decision.vote_detail["risk_model"]["meta_size_mult"] = size_mult
                        if widen_mult > 1.0 and final_decision.vote_detail["risk_model"].get("stop_loss"):
                            final_decision.vote_detail["risk_model"]["stop_loss"] = final_decision.vote_detail["risk_model"]["stop_loss"] * widen_mult
                            final_decision.vote_detail["risk_model"]["meta_stop_widen"] = widen_mult
                except Exception:
                    pass

                # Sprint 30: MTC partial dampen (apply AFTER quality scaling, BEFORE liquidity)
                try:
                    mtc = final_decision.vote_detail.get("mtc_gate")
                    if mtc and mtc.get("action") == "DAMPEN" and final_decision.vote_detail.get("risk_model"):
                        mtc_cfg = (self.settings.get("mtc") or {}).get("actions", {})
                        partial_cfg = (mtc_cfg.get("partial") or {})
                        size_mult = float(partial_cfg.get("size_mult", 0.6))
                        widen_mult = float(partial_cfg.get("widen_stop_mult", 1.10))
                        if 0 < size_mult < 1.0:
                            base_sz = final_decision.vote_detail["risk_model"]["position_size"]
                            final_decision.vote_detail["risk_model"]["position_size"] = round(base_sz * size_mult, 2)
                            final_decision.vote_detail["risk_model"]["mtc_scaled"] = True
                            final_decision.vote_detail["risk_model"]["mtc_size_mult"] = size_mult
                        if widen_mult > 1.0 and final_decision.vote_detail["risk_model"].get("stop_loss"):
                            final_decision.vote_detail["risk_model"]["stop_loss"] = final_decision.vote_detail["risk_model"]["stop_loss"] * widen_mult
                            final_decision.vote_detail["risk_model"]["mtc_stop_widen"] = widen_mult
                except Exception:
                    pass

                # Apply liquidity gate dampen AFTER quality scaling but BEFORE event gate dampen (multiplicative layering)
                try:
                    lq = final_decision.vote_detail.get("liquidity_gate")
                    if lq and lq.get("action") == "DAMPEN" and final_decision.vote_detail.get("risk_model"):
                        mult = float(lq.get("size_mult") or 1.0)
                        if 0 < mult < 1.0:
                            base_sz = final_decision.vote_detail["risk_model"]["position_size"]
                            final_decision.vote_detail["risk_model"]["position_size"] = round(base_sz * mult, 2)
                            final_decision.vote_detail["risk_model"]["lq_gate_scaled"] = True
                            final_decision.vote_detail["risk_model"]["lq_size_mult"] = mult
                        # widen stop
                        widen = float(lq.get("widen_stop_mult") or 1.0)
                        if widen > 1.0 and final_decision.vote_detail["risk_model"].get("stop_loss"):
                            final_decision.vote_detail["risk_model"]["stop_loss"] = final_decision.vote_detail["risk_model"]["stop_loss"] * widen
                            final_decision.vote_detail["risk_model"]["lq_stop_widen"] = widen
                        # maker-only propagation (adjust playbook/plan if exists)
                        if lq.get("maker_only") and isinstance(final_decision.vote_detail.get("playbook"), dict):
                            try:
                                pb = final_decision.vote_detail.get("playbook")
                                if pb.get("entry_type") == "market":
                                    pb["entry_type"] = "maker"
                                    pb["maker_from_lq_gate"] = True
                            except Exception:
                                pass
                    elif lq and lq.get("maker_only") and isinstance(final_decision.vote_detail.get("playbook"), dict):
                        # Even if action NONE but maker_only flag (future config) -> propagate
                        try:
                            pb = final_decision.vote_detail.get("playbook")
                            if pb.get("entry_type") == "market":
                                pb["entry_type"] = "maker"
                                pb["maker_from_lq_gate"] = True
                        except Exception:
                            pass
                except Exception:
                    pass

                # Apply whale gate size modulation AFTER liquidity gate scaling (multiplicative) but BEFORE event gate scaling
                try:
                    wg = final_decision.vote_detail.get('whale_gate')
                    if wg and wg.get('action') in ('DAMPEN','BOOST') and final_decision.vote_detail.get('risk_model'):
                        mult = float(wg.get('size_mult') or 1.0)
                        # Allow >1 (boost) or <1 (dampen), guard extremes
                        if mult > 0 and mult != 1.0:
                            base_sz = final_decision.vote_detail['risk_model']['position_size']
                            capped_mult = min(mult, 2.5)  # cap boost to 2.5x for safety
                            final_decision.vote_detail['risk_model']['position_size'] = round(base_sz * capped_mult, 2)
                            final_decision.vote_detail['risk_model']['whale_scaled'] = True
                            final_decision.vote_detail['risk_model']['whale_size_mult'] = capped_mult
                except Exception:
                    pass

                # Apply event gate dampen AFTER quality scaling (so multiplicative)
                try:
                    eg = final_decision.vote_detail.get('event_gate')
                    if eg and eg.get('action') == 'DAMPEN' and final_decision.vote_detail.get('risk_model'):
                        mult = float(eg.get('size_mult') or 1.0)
                        if 0 < mult < 1.0:
                            base_sz = final_decision.vote_detail['risk_model']['position_size']
                            final_decision.vote_detail['risk_model']['position_size'] = round(base_sz * mult, 2)
                            final_decision.vote_detail['risk_model']['event_gate_scaled'] = True
                            final_decision.vote_detail['risk_model']['event_gate_mult'] = mult
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
                                # --- Sprint 53: Liquidity-Adjusted VaR computation and exec hint ---
                                try:
                                    if getattr(self, '_lvar_engine', None) is not None and getattr(self, '_exec_adapter', None) is not None:
                                        price_now = float(feature_vector.ohlcv.get('close', 0.0))
                                        # notional and qty for proposed size
                                        proposed_notional = float(final_decision.vote_detail["position_sizer"]["size_quote"]) or 0.0
                                        q = proposed_notional / (price_now or 1.0)
                                        # ADV best-effort: try volumetric features or settings
                                        adv = float((self.settings.get('market') or {}).get('adv_usd', 0.0) or 0.0)
                                        # book depth best-effort from FeatureStore
                                        try:
                                            bd = self.feature_store.get_book_ticker(symbol)
                                            book_depth = float((bd[1] if bd and len(bd)>1 else 0.0) + (bd[3] if bd and len(bd)>3 else 0.0)) if bd else 0.0
                                        except Exception:
                                            book_depth = 0.0
                                        # lambda estimate
                                        lam = None
                                        try:
                                            lam = self.feature_store.get_lambda_for(symbol)
                                        except Exception:
                                            lam = None
                                        # sigma / vol forecast
                                        sigma = None
                                        try:
                                            vol_obj = features.get('volatility') if isinstance(features, dict) else None
                                            sigma = float(getattr(vol_obj, 'sigma', None) or getattr(vol_obj, 'atr', None) or 0.0)
                                        except Exception:
                                            sigma = 0.0
                                        # z_alpha (VaR quantile) from settings
                                        z_alpha = float((self.settings.get('risk') or {}).get('var_z', 2.33))
                                        pr = float((self.settings.get('risk') or {}).get('participation_rate', 0.12))
                                        lvar_out = self._lvar_engine.compute(sigma=sigma, z_alpha=z_alpha, notional=proposed_notional, price=price_now, q=q, adv=adv, pr=pr, book_depth=book_depth, lam=lam, stress_multiplier=float((self.settings.get('risk') or {}).get('stress_mult', 1.0)))
                                        # attach feature view for downstream transports
                                        final_decision.vote_detail.setdefault('risk_liquidity', {})
                                        final_decision.vote_detail['risk_liquidity'].update({
                                            'lvar_$': round(lvar_out.lvar_usd, 2),
                                            'lvar_pct_equity': round(lvar_out.lvar_pct_equity, 6),
                                            'liq_cost_$': round(lvar_out.liq_cost_usd, 2),
                                            'ttl_minutes': round(lvar_out.ttl_minutes, 2),
                                            'stress_factor': lvar_out.stress_factor,
                                        })
                                        # Exec suggestion
                                        sug = self._exec_adapter.suggest(lvar_pct=lvar_out.lvar_pct_equity, liq_cost_pct=(lvar_out.liq_cost_usd / max(1e-12, float(equity))) if hasattr(self,'_lvar_engine') else 0.0, ttl_minutes=lvar_out.ttl_minutes)
                                        final_decision.vote_detail['risk_liquidity'].update({'size_suggested_mult': sug.size_multiplier, 'exec_style_hint': sug.exec_style, 'exec_reason': sug.reason})
                                        # apply suggestion multiplicatively to position_size
                                        try:
                                            if sug.size_multiplier == 0.0:
                                                # veto trade
                                                final_decision.decision = 'FLAT'
                                                final_decision.vetoes.append('LVAR_VETO')
                                                final_decision.vote_detail.setdefault('reason', 'LVAR_VETO')
                                            else:
                                                if final_decision.vote_detail.get('position_sizer') and final_decision.vote_detail['position_sizer'].get('size_quote'):
                                                    base_sz = float(final_decision.vote_detail['position_sizer']['size_quote'])
                                                    new_sz = round(base_sz * float(sug.size_multiplier), 2)
                                                    final_decision.vote_detail['position_sizer']['size_quote'] = new_sz
                                                    final_decision.vote_detail['position_sizer']['lvar_scaled'] = True
                                                    final_decision.vote_detail['position_sizer']['lvar_size_mult'] = sug.size_multiplier
                                        except Exception:
                                            pass
                                        # Persist risk_liquidity row into FeatureStore macro export buffer for diagnostics
                                        try:
                                            # Only emit if feature-store diagnostics are enabled in settings to avoid overhead
                                            emit_diag = bool((self.settings.get('cross_asset', {}) or {}).get('diagnostics', {}).get('emit', False))
                                            if emit_diag and getattr(self, 'feature_store', None) is not None:
                                                try:
                                                    rl = final_decision.vote_detail.get('risk_liquidity') or {}
                                                    row = {
                                                        'symbol': symbol,
                                                        'ts': ts_epoch,
                                                        'lvar_usd': rl.get('lvar_$'),
                                                        'lvar_pct_equity': rl.get('lvar_pct_equity'),
                                                        'liq_cost_usd': rl.get('liq_cost_$'),
                                                        'ttl_minutes': rl.get('ttl_minutes'),
                                                        'stress_factor': rl.get('stress_factor'),
                                                        'size_suggested_mult': rl.get('size_suggested_mult'),
                                                        'exec_style_hint': rl.get('exec_style_hint'),
                                                        'exec_reason': rl.get('exec_reason')
                                                    }
                                                    # best-effort call into internal buffer helper (FeatureStore exposes this internally)
                                                    if hasattr(self.feature_store, '_buffer_macro_row'):
                                                        try:
                                                            self.feature_store._buffer_macro_row(row)
                                                        except Exception:
                                                            # swallow to avoid breaking engine
                                                            pass
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
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

                                # Apply meta gate dampen to position_sizer result (multiplicative) if present
                                try:
                                    mg = final_decision.vote_detail.get("meta_gate")
                                    if mg and mg.get("action") == "DAMPEN":
                                        size_mult = float(mg.get("size_mult") or 1.0)
                                        if 0 < size_mult < 1.0:
                                            base_sz = final_decision.vote_detail["position_sizer"]["size_quote"]
                                            final_decision.vote_detail["position_sizer"]["size_quote"] = round(base_sz * size_mult, 2)
                                            final_decision.vote_detail["position_sizer"]["meta_scaled"] = True
                                            final_decision.vote_detail["position_sizer"]["meta_size_mult"] = size_mult
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # Sprint 32 Advanced Sizer (after legacy risk_model & overlays so it can override quantity logic cleanly)
                try:
                    sizer_cfg = (self.settings.get('sizer') or {}) if isinstance(self.settings, dict) else {}
                    if getattr(self._adv_sizer, 'enabled', False) and final_decision.decision in ("LONG","SHORT") and sizer_cfg.get('enabled', True):
                        # collect inputs
                        p_meta = None
                        try:
                            mg = final_decision.vote_detail.get('meta_gate')
                            if mg:
                                p_meta = mg.get('p')
                        except Exception:
                            pass
                        mtc_status = None
                        try:
                            mtc = final_decision.vote_detail.get('mtc_gate')
                            if mtc:
                                mtc_status = mtc.get('status')
                        except Exception:
                            pass
                        lq_action = None
                        try:
                            lq = final_decision.vote_detail.get('liquidity_gate')
                            if lq:
                                lq_action = lq.get('action')
                        except Exception:
                            pass
                        atr_val = None
                        try:
                            vol_obj = features.get('volatility')
                            if vol_obj:
                                atr_val = getattr(vol_obj,'atr', None)
                        except Exception:
                            pass
                        # drawdown placeholder (need equity peak tracking; use 0 for now)
                        drawdown = 0.0
                        adv_res = self._adv_sizer.compute(symbol, final_decision.decision, float(feature_vector.ohlcv.get('close',0.0)), equity, {
                            'p_meta': p_meta,
                            'mtc_status': mtc_status,
                            'liquidity_gate_action': lq_action,
                            'atr': atr_val,
                            'stop_distance': None,
                            'drawdown': drawdown,
                            'open_positions': [],  # live path lacks portfolio book; integrate later
                        })
                        final_decision.vote_detail.setdefault('advanced_sizer', adv_res.breakdown)
                        # Record advisory sizing fields
                        if final_decision.vote_detail.get('risk_model'):
                            price_cur = float(feature_vector.ohlcv.get('close',0.0))
                            notional = adv_res.qty * price_cur if adv_res.qty>0 else 0.0
                            rm = final_decision.vote_detail['risk_model']
                            rm['position_size_adv'] = round(notional,2)
                            rm['adv_qty'] = adv_res.qty
                            rm['adv_risk_pct'] = adv_res.risk_pct_effective
                        # Optional live enforcement toggle
                        if sizer_cfg.get('enforce_live'):
                            if adv_res.qty <= 0:
                                # veto trade (size zero)
                                final_decision.decision = 'FLAT'
                                final_decision.vetoes.append('ADV_SIZER_ZERO')
                                final_decision.vote_detail.setdefault('reason','ADV_SIZER_ZERO')
                            else:
                                # Override existing risk_model position_size with advanced sizing notional
                                if final_decision.vote_detail.get('risk_model'):
                                    final_decision.vote_detail['risk_model']['position_size'] = round(adv_res.qty * float(feature_vector.ohlcv.get('close',0.0)),2)
                                    final_decision.vote_detail['risk_model']['enforced_adv'] = True
                except Exception:
                    pass
        except Exception as e:
            logger.exception("Adaptive sizing error: {}", e)

        # ---------------- Sprint 33 Portfolio Risk Allocation (pre-beta gating but after advanced sizing) ----------------
        try:
            pr_cfg = (self.settings.get('portfolio_risk') or {}) if isinstance(self.settings, dict) else {}
            if pr_cfg.get('enabled') and RiskEstimator and PortfolioAllocator and final_decision.decision in ("LONG","SHORT"):
                # Lazy init estimator & allocator
                if not hasattr(self, '_pr_est'):
                    self._pr_est = RiskEstimator(self.settings)
                    self._pr_alloc = PortfolioAllocator(self.settings, self._pr_est)
                    self._pr_bar_counter = 0
                # Update estimator with this bar (needs high/low/close). Use ohlcv_segment last row.
                try:
                    bar_row = ohlcv_segment.iloc[-1].to_dict()
                    self._pr_est.update(symbol, bar_row, ts_epoch)
                except Exception:
                    pass
                # Build candidate structure if advanced_sizer produced risk_amount
                adv = final_decision.vote_detail.get('advanced_sizer') if isinstance(final_decision.vote_detail, dict) else {}
                risk_amt = float(adv.get('risk_amount', 0.0)) if adv else 0.0
                stop_dist = float(adv.get('stop_distance', 0.0)) if adv else None
                price_now = float(feature_vector.ohlcv.get('close',0.0)) if feature_vector and feature_vector.ohlcv else 0.0
                qty = float(adv.get('qty', 0.0)) if adv else 0.0
                candidate = {
                    'symbol': symbol,
                    'side': final_decision.decision,
                    'risk_amount': risk_amt,
                    'stop_distance': stop_dist or 0.0,
                    'price': price_now,
                    'qty': qty,
                }
                # TODO: incorporate real open positions book; placeholder empty list for now
                open_positions = getattr(self, '_external_open_positions', []) or []
                adjustments, metrics = self._pr_alloc.evaluate(open_positions, candidate, ts_epoch)
                if adjustments:
                    for adj in adjustments:
                        if adj['symbol'] == symbol:
                            if adj['action'] == 'reject':
                                final_decision.decision = 'FLAT'
                                final_decision.vetoes.append('PR_VETO')
                                final_decision.vote_detail.setdefault('reason','PR_VETO')
                            elif adj['action'] == 'scale' and adj['size_mult']>0 and final_decision.vote_detail.get('risk_model'):
                                try:
                                    rm = final_decision.vote_detail['risk_model']
                                    rm['position_size'] = round(rm['position_size'] * adj['size_mult'],2)
                                    rm['pr_scaled'] = True
                                    rm['pr_size_mult'] = adj['size_mult']
                                except Exception:
                                    pass
                final_decision.vote_detail.setdefault('portfolio_risk', {})
                final_decision.vote_detail['portfolio_risk'].update({'adjustments': adjustments, 'metrics': metrics})
                # Record metrics timeseries if present
                if metrics:
                    row = dict(metrics)
                    row['ts'] = ts_epoch
                    row['symbol'] = symbol
                    try:
                        self._pr_metrics_ts.append(row)
                    except Exception:
                        pass
        except Exception as e:  # pragma: no cover
            logger.debug('Portfolio risk allocation error: {}', e)

        # Sprint 22: Pre-trade beta/cluster gating (only if decision still active LONG/SHORT)
        if self._ph_enabled and final_decision and final_decision.decision in ("LONG", "SHORT"):
            try:
                ph_cfg = self.settings.get("portfolio_hedge", {}) if isinstance(self.settings, dict) else {}
                # proposed notional from risk_model / position_sizer
                notional = None
                rm = final_decision.vote_detail.get("risk_model") if isinstance(final_decision.vote_detail, dict) else None
                if rm:
                    notional = rm.get("position_size")
                if notional is None:
                    ps = final_decision.vote_detail.get("position_sizer") if isinstance(final_decision.vote_detail, dict) else None
                    if ps:
                        notional = ps.get("size_quote")
                if notional is None:
                    notional = 0.01 * self._ph_equity  # fallback
                side_sign = 1.0 if final_decision.decision == "LONG" else -1.0
                preview = self._ph_caps.preview_beta_after_trade(
                    symbol=symbol,
                    add_notional=side_sign * float(notional),
                    equity=self._ph_equity,
                    exposure_symbols=self._ph_exposure.symbol_notionals,
                    betas=self._ph_corr.betas,
                    cluster_map=self._ph_exposure.cluster_map,
                )
                cluster_veto = self._ph_caps.check_cluster_caps(
                    symbol=symbol,
                    add_notional=side_sign * float(notional),
                    equity=self._ph_equity,
                    exposure_symbols=self._ph_exposure.symbol_notionals,
                    cluster_map=self._ph_exposure.cluster_map,
                )
                if cluster_veto:
                    final_decision.decision = "FLAT"
                    final_decision.vetoes.append("VETO_CLUSTER_CAP")
                    final_decision.vote_detail.setdefault("reason", "VETO_CLUSTER_CAP")
                elif not preview.allowed:
                    final_decision.decision = "FLAT"
                    if preview.veto_reason:
                        final_decision.vetoes.append("VETO_" + preview.veto_reason)
                        final_decision.vote_detail.setdefault("reason", preview.veto_reason)
                else:
                    if abs(preview.scaled_notional - side_sign * float(notional)) > 1e-6:
                        scale = abs(preview.scaled_notional) / (abs(float(notional)) or 1.0)
                        if rm and rm.get("position_size"):
                            rm["position_size"] = round(float(rm["position_size"]) * scale, 2)
                            rm["beta_scaled"] = True
                        ps = final_decision.vote_detail.get("position_sizer")
                        if ps and ps.get("size_quote"):
                            ps["size_quote"] = round(float(ps["size_quote"]) * scale, 2)
                            ps["beta_scaled"] = True
                        final_decision.vote_detail.setdefault("beta_preview", {})
                        final_decision.vote_detail["beta_preview"].update({"projected_beta": round(preview.projected_beta, 4), "scaled": True})
                    else:
                        final_decision.vote_detail.setdefault("beta_preview", {})
                        final_decision.vote_detail["beta_preview"].update({"projected_beta": round(preview.projected_beta, 4), "scaled": False})
            except Exception as e:  # pragma: no cover
                logger.debug("Portfolio beta gating error: {}", e)

        # Sprint 22: Hedge plan (informational until execution layer consumes)
        if self._ph_enabled:
            try:
                beta_p = self._ph_corr.portfolio_beta(self._ph_exposure.symbol_notionals, self._ph_equity)
                # dynamic beta target (leader bias) if configured
                beta_target = 0.0
                try:
                    lb = getattr(self, "_ph_bias_cfg", {}) or {}
                    if lb.get("enabled", False):
                        # Use correlation regime + leader trend slope for directional bias
                        if self._ph_corr.high_corr_regime:
                            # leader features
                            leader_feats = features if symbol == self._ph_corr.leader else self._get_features_robust(self._ph_corr.leader, timestamp, tf)
                            trend_block = leader_feats.get("trend") if isinstance(leader_feats, dict) else None
                            slope = None
                            try:
                                slope = getattr(trend_block, "slope", None) or getattr(trend_block, "trend_slope", None)
                            except Exception:
                                slope = None
                            try:
                                slope = float(slope) if slope is not None else 0.0
                            except Exception:
                                slope = 0.0
                            bias_long = float(lb.get("bias_long", 0.05))
                            bias_short = float(lb.get("bias_short", -0.05))
                            if slope and slope > 0:
                                beta_target = bias_long
                            elif slope and slope < 0:
                                beta_target = bias_short
                except Exception:
                    beta_target = 0.0
                plan = self._ph_hedger.compute_plan(bar_index=int(ts_epoch), portfolio_beta=beta_p, equity=self._ph_equity, beta_target=beta_target)
                final_decision.vote_detail.setdefault("hedger", {})
                final_decision.vote_detail["hedger"].update({
                    "beta_p": round(beta_p, 4),
                    "action": plan.action,
                    "reason": plan.reason,
                    "hedge_notional_target": round(plan.target_notional, 2),
                    "beta_target": round(beta_target, 4),
                })
                self._ph_report.record(HedgeSnapshot(ts=int(ts_epoch), beta_p=beta_p, hedge_notional=self._ph_hedger.current_hedge_notional, action=plan.action, reason=plan.reason))
            except Exception as e:  # pragma: no cover
                logger.debug("Hedge plan compute error: {}", e)

        # High-correlation quality gate (simplified): if regime high correlation AND projected beta would push further outside band AND confidence low -> veto
        if self._ph_enabled and hasattr(self, "_ph_corr") and self._ph_corr.high_corr_regime:
            try:
                hedger_band = getattr(self._ph_hedger, "beta_band", (-0.15, 0.15))
                beta_p_now = self._ph_corr.portfolio_beta(self._ph_exposure.symbol_notionals, self._ph_equity)
                conf = float(getattr(final_decision, "confidence", 0.0) or 0.0)
                # If low confidence (<0.25) and already outside 120% of band width in direction of trade, veto
                band_min, band_max = hedger_band
                band_width = band_max - band_min
                if conf < 0.25:
                    if final_decision.decision == "LONG" and beta_p_now > band_max + 0.2 * band_width:
                        final_decision.vetoes.append("VETO_HIGH_CORR_QUALITY")
                        final_decision.decision = "FLAT"
                        final_decision.vote_detail.setdefault("reason", "HIGH_CORR_QG")
                    elif final_decision.decision == "SHORT" and beta_p_now < band_min - 0.2 * band_width:
                        final_decision.vetoes.append("VETO_HIGH_CORR_QUALITY")
                        final_decision.decision = "FLAT"
                        final_decision.vote_detail.setdefault("reason", "HIGH_CORR_QG")
            except Exception:
                pass

        # --- NEW: Drift policy evaluation & telemetry / audit hooks ---
        try:
            if getattr(self, '_policy_engine', None) is not None and final_decision is not None:
                # Build compact metrics snapshot from vote_detail
                vd = final_decision.vote_detail if isinstance(final_decision.vote_detail, dict) else {}
                metrics = {
                    'sprt_state': vd.get('sprt', {}).get('state') if vd.get('sprt') else vd.get('sprt_state'),
                    'pf_delta_pct': vd.get('pf_delta_pct') or vd.get('pf_delta') or 0.0,
                    'maxdd_p95_breach': vd.get('maxdd_p95_breach') or vd.get('maxdd_breach') or False,
                    'ece_live': vd.get('ece') or vd.get('ece_live') or 0.0,
                    'slip_delta_bps': vd.get('slip_bps') or vd.get('slip_delta_bps') or 0.0,
                }
                action = self._policy_engine.evaluate(metrics)
                # Emit telemetry if telemetry logger present in settings injected objects
                try:
                    tel = (self.settings.get('_telemetry') or None)
                    if tel and hasattr(tel, 'emit_policy_action'):
                        tel.emit_policy_action(final_decision.symbol if hasattr(final_decision,'symbol') else symbol, 0, {'type': action.type.value if hasattr(action,'type') else str(action.type), 'size_mult': action.size_mult, 'reason_codes': list(action.reason_codes), 'timestamp': action.timestamp}, metrics)
                except Exception:
                    pass
                # Persist audit record (best-effort) into DB
                try:
                    from ultra_signals.persist.db import record_policy_action, write_retrain_job
                    ts_rec = int(final_decision.ts or ts_epoch)
                    record_policy_action(final_decision.symbol, ts_rec, action.type.value if hasattr(action,'type') else str(action.type), float(action.size_mult or 1.0), list(action.reason_codes) if getattr(action,'reason_codes',None) else None, metrics)
                    # If retrain requested, write job to configured queue_dir
                    if getattr(action, 'type', None) and (str(action.type).lower() == 'retrain' or (hasattr(action,'type') and action.type == action.type.RETRAIN)):
                        qdir = (self.settings.get('retrain') or {}).get('queue_dir') if isinstance(self.settings, dict) else None
                        job = {
                            'model': (self.settings.get('model_id') or 'alpha_model'),
                            'symbol': final_decision.symbol,
                            'trigger_ts': ts_rec,
                            'metrics': metrics,
                            'reason_codes': list(action.reason_codes) if getattr(action,'reason_codes',None) else []
                        }
                        if qdir:
                            write_retrain_job(qdir, job)
                except Exception:
                    pass
        except Exception:
            pass

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
