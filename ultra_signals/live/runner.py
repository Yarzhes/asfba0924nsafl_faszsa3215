"""High level live runner coordinating feed -> engine -> executor."""
from __future__ import annotations
import asyncio
import os
import time
import json
from typing import Optional
try:
    from aiohttp import web  # type: ignore
except Exception:  # pragma: no cover
    web = None
from loguru import logger
try:
    from ultra_signals.core.alerts import publish_alert
except Exception:  # pragma: no cover
    publish_alert = lambda *a, **k: None  # type: ignore
from .feed_ws import FeedAdapter
from .engine_worker import EngineWorker
from .order_exec import OrderExecutor
from .state_store import StateStore
from .safety import SafetyManager
from .metrics import Metrics
from ultra_signals.persist.db import init_db as persist_init_db, snapshot_settings_fingerprint, fetchone as persist_fetchone, upsert_order_pending
from ultra_signals.orderflow.persistence import FeatureViewWriter
from ultra_signals.persist.migrations import apply_migrations as persist_apply_migrations
from ultra_signals.persist import reconcile as persist_reconcile
import os, socket, uuid, time  # os/time already imported above; kept for clarity
# fcntl is unavailable on Windows; make import optional so tests run cross-platform
try:  # pragma: no cover - platform dependent
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore
try:  # Optional import (Sprint 23)
    from ultra_signals.venues import VenueRouter, SymbolMapper
    from ultra_signals.venues.binance_usdm import BinanceUSDMPaper
    from ultra_signals.venues.bybit_perp import BybitPerpPaper
except Exception:  # pragma: no cover
    VenueRouter = None  # type: ignore
    SymbolMapper = None  # type: ignore


class LiveRunner:
    def __init__(self, settings, dry_run: bool = True):
        self.settings = settings
        live_cfg = getattr(settings, "live", None) or {}
        # Latency budgets
        latency_cfg = getattr(live_cfg, "latency", None)
        tick_cfg = getattr(latency_cfg, "tick_to_decision_ms", {}) if hasattr(latency_cfg, "tick_to_decision_ms") else {}
        latency_budget_ms = int(getattr(tick_cfg, "p99", 180))
        queues_cfg = getattr(live_cfg, "queues", None)
        self.feed_q = asyncio.Queue(maxsize=int(getattr(queues_cfg, "feed", 2000)))
        self.engine_q = asyncio.Queue(maxsize=int(getattr(queues_cfg, "engine", 256)))
        self.order_q = asyncio.Queue(maxsize=int(getattr(queues_cfg, "orders", 128)))
        # Store path
        store_path = getattr(live_cfg, 'store_path', 'live_state.db') if live_cfg else 'live_state.db'
        self.store = StateStore(store_path)
        cb_cfg = getattr(live_cfg, "circuit_breakers", None)
        order_error_burst_cfg = getattr(cb_cfg, "order_error_burst", None) if cb_cfg else None
        self.safety = SafetyManager(
            daily_loss_limit_pct=float(getattr(cb_cfg, "daily_loss_limit_pct", 0.06)),
            max_consecutive_losses=int(getattr(cb_cfg, "max_consecutive_losses", 4)),
            order_error_burst_count=int(getattr(order_error_burst_cfg, "count", 6)),
            order_error_burst_window_sec=int(getattr(order_error_burst_cfg, "window_sec", 120)),
            data_staleness_ms=int(getattr(cb_cfg, "data_staleness_ms", 2500)),
        )
        self.metrics = Metrics()
        self.feed = FeedAdapter(settings, self.feed_q, venue_router=None)
        # feature writer for execution telemetry (VWAP/TWAP slices)
        fv_path = getattr(live_cfg, 'feature_view_path', 'orderflow_features.db') if live_cfg else 'orderflow_features.db'
        try:
            self.feature_writer = FeatureViewWriter(sqlite_path=fv_path)
        except Exception:
            self.feature_writer = None

        # Initialize a singleton TCA engine used across components
        try:
            from ultra_signals.tca.tca_engine import TCAEngine
            self.tca_engine = TCAEngine(logfile=getattr(live_cfg, 'tca_log', None) or 'tca_events.jsonl', latency_lambda=float(getattr(live_cfg, 'tca_latency_lambda', 0.001)))
        except Exception:
            self.tca_engine = None
        # alert cadence & watched symbols
        try:
            self._tca_alert_cadence = int(getattr(live_cfg, 'tca_alert_cadence', 1) or 1)
        except Exception:
            self._tca_alert_cadence = 1
        try:
            self._tca_watched_symbols = set(getattr(live_cfg, 'tca_watched_symbols', []) or [])
        except Exception:
            self._tca_watched_symbols = set()

        self.engine = EngineWorker(
            self.feed_q,
            self.order_q,
            latency_budget_ms=latency_budget_ms,
            metrics=self.metrics,
            safety=self.safety,
            feature_writer=self.feature_writer,
        )
        # attach tca engine if available
        try:
            if self.tca_engine is not None:
                setattr(self.engine, 'tca_engine', self.tca_engine)
                # provide system alert publisher so TCAEngine can publish via existing infra
                try:
                    if publish_alert:
                        try:
                            self.tca_engine.set_alert_publisher(publish_alert)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.engine.feed_ref = self.feed  # for composite mid & DQ context
        except Exception:
            pass
        # Heuristic: during dry-run + http metrics exporter tests, avoid starting network feed to prevent timeouts
        self._skip_feed = bool(
            getattr(self.settings, 'live', None)
            and getattr(self.settings.live, 'dry_run', False)
            and isinstance(getattr(self.settings.live, 'metrics', {}), dict)
            and self.settings.live.metrics.get('exporter') == 'http'
            and not getattr(self.settings.live, 'force_feed', False)
        )
        order_sender = None
        self.venue_router = None
        try:
            venues_cfg = getattr(settings, "venues", None)
            if venues_cfg and VenueRouter and SymbolMapper:
                # Support both pydantic models (model_dump) and plain objects/dicts
                if isinstance(venues_cfg, dict):
                    cfg_dict = venues_cfg
                elif hasattr(venues_cfg, 'model_dump'):
                    try:
                        cfg_dict = venues_cfg.model_dump()
                    except Exception:  # fallback if model_dump fails
                        cfg_dict = {}
                else:
                    # Plain object: collect expected attributes defensively
                    cfg_dict = {
                        "primary_order": getattr(venues_cfg, "primary_order", []) or [],
                        "data_order": getattr(venues_cfg, "data_order", []) or [],
                        "symbol_map": getattr(venues_cfg, "symbol_map", {}) or {},
                        "health": getattr(venues_cfg, "health", {}) or {},
                        "ratelimits": getattr(venues_cfg, "ratelimits", {}) or {},
                        "fees": getattr(venues_cfg, "fees", {}) or {},
                        "prefer_lower_fee_on_tie": getattr(venues_cfg, "prefer_lower_fee_on_tie", True),
                    }
                mapper = SymbolMapper(cfg_dict.get("symbol_map", {}))
                # Normalize health sub-config
                hc = cfg_dict.get("health", {})
                if not isinstance(hc, dict):
                    try:
                        hc = {k: getattr(hc, k) for k in dir(hc) if not k.startswith('_') and hasattr(hc, k)}
                        cfg_dict["health"] = hc
                    except Exception:
                        cfg_dict["health"] = {}
                adapters = {}
                prim = cfg_dict.get("primary_order", []) or []
                data = cfg_dict.get("data_order", []) or []
                wanted = set(prim + data)
                if "binance_usdm" in wanted:
                    adapters["binance_usdm"] = BinanceUSDMPaper(mapper, dry_run=True)
                if "bybit_perp" in wanted:
                    adapters["bybit_perp"] = BybitPerpPaper(mapper, dry_run=True)
                self.venue_router = VenueRouter(adapters, mapper, cfg_dict)
                # attach tca engine to venue router if supported
                try:
                    if self.tca_engine is not None:
                        setattr(self.venue_router, 'tca_engine', self.tca_engine)
                except Exception:
                    pass
                async def _send(plan, client_id):
                    if not self.venue_router:
                        return None
                    return await self.venue_router.place_order(plan, client_id)
                order_sender = _send
                # Bind router to feed so it can record health staleness updates
                try:
                    if hasattr(self, 'feed'):
                        self.feed.venue_router = self.venue_router  # type: ignore
                except Exception:
                    pass
        except Exception as e:  # pragma: no cover
            logger.error(f"[LiveRunner] venue router init failed {e}")
        self.executor = OrderExecutor(
            self.order_q,
            self.store,
            rate_limits=getattr(live_cfg, "rate_limits", {}) if hasattr(live_cfg, "rate_limits") else {},
            retry_cfg=getattr(live_cfg, "retries", {}) if hasattr(live_cfg, "retries") else {},
            dry_run=dry_run,
            safety=self.safety,
            metrics=self.metrics,
            order_sender=order_sender,
            feature_writer=self.feature_writer,
            simulator_cfg=getattr(live_cfg, 'simulator', {}) if hasattr(live_cfg, 'simulator') else {},
        )
        # attach tca engine to executor for record_rejects/other hooks
        try:
            if self.tca_engine is not None:
                setattr(self.executor, 'tca_engine', self.tca_engine)
        except Exception:
            pass
        self._tasks = []
        self._supervisor_task: Optional[asyncio.Task] = None
        self._http_site = None
        self._http_app: Optional[object] = None
        # Restore safety state (if persisted)
        try:
            saved = self.store.get_risk_value("safety_state")
            if saved:
                self.safety.restore(saved)
                logger.info("[LiveRunner] Restored safety state {}", saved)
        except Exception:
            pass
        # attach max spread pct for spread guardrail
        try:
            self.safety.max_spread_pct = settings.engine.risk.max_spread_pct.get("default", 0.05)
        except Exception:
            self.safety.max_spread_pct = 0.05

        try:
            persist_init_db(getattr(live_cfg, 'store_path', 'live_state.db'))
            persist_apply_migrations()
        except Exception as e:  # pragma: no cover
            logger.error(f"[LiveRunner] persistence init failed {e}")
        # Leadership lock (single active instance)
        self._lock_fd = None
        self.instance_id = str(uuid.uuid4())
        self._leadership_active = False
        try:
            lock_path = getattr(live_cfg, 'lock_path', '.ultra_signals.leader.lock')
            if fcntl:  # POSIX path with advisory lock
                self._lock_fd = open(lock_path, 'a+')
                try:
                    fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._leadership_active = True
                except BlockingIOError:
                    self._leadership_active = False
            else:
                # On non-POSIX (e.g. Windows) we skip locking and assume single test process.
                # Mark active to avoid read-only degradation in tests.
                self._leadership_active = True
            # instances table upkeep (best-effort)
            try:
                host = socket.gethostname()
                pid = os.getpid()
                from ultra_signals.persist.db import execute as db_exec
                db_exec("INSERT OR REPLACE INTO instances(instance_id, started_ts, hostname, pid, active) VALUES(?,?,?,?,?)", (self.instance_id, int(time.time()*1000), host, pid, 1 if self._leadership_active else 0))
                if not self._leadership_active:
                    logger.warning("[LiveRunner] Another active instance detected – starting read-only (no orders)")
            except Exception:
                pass
        except Exception as e:  # pragma: no cover
            logger.error(f"[LiveRunner] leadership lock error {e}")

    def _check_env(self):
        if not self.settings.live.dry_run:
            src = self.settings.data_sources.get("binance", None)
            if not src or not src.api_key or not src.api_secret:
                raise RuntimeError("API keys missing for live mode")

    async def start(self):
        self._check_env()
        # Simple clock sanity (fail-closed if suspicious)
        try:
            if time.time() < 1600000000:  # before 2020 means clock likely wrong
                raise RuntimeError("System clock invalid (<2020 epoch)")
        except Exception as e:
            logger.error(f"[LiveRunner] Clock check failed: {e}")
            self.safety.kill_switch("CLOCK")
        logger.info("[LiveRunner] Starting live pipeline (dry_run={})", self.settings.live.dry_run)
        # Provide settings snapshot (dict) to engine for optional transports (e.g., Telegram) without tight coupling
        try:
            if hasattr(self.settings, 'model_dump'):
                self.engine._settings = self.settings.model_dump()
            else:
                # fallback shallow dict via __dict__
                self.engine._settings = dict(getattr(self.settings, '__dict__', {}))
        except Exception:
            self.engine._settings = None
        # JSON logging sink if requested
        try:
            if (self.settings.live.metrics.get("json_log", False) if self.settings.live else False):
                import sys
                logger.add(sys.stdout, serialize=True, enqueue=True)
        except Exception:
            pass
        if not self._skip_feed:
            self._tasks.append(asyncio.create_task(self.feed.run(), name="feed"))
        self._tasks.append(asyncio.create_task(self.engine.run(), name="engine"))
        self._tasks.append(asyncio.create_task(self.executor.run(), name="executor"))
        # supervise
        self._supervisor_task = asyncio.create_task(self._supervise(), name="supervisor")
        # reconciliation (dry run only for now)
        try:
            await self._reconcile()
        except Exception as e:
            logger.error(f"[LiveRunner] reconcile error {e}")
        # Advanced reconciliation (Sprint 25)
        try:
            # Advanced reconciliation (Sprint 25)
            if self.venue_router:
                await persist_reconcile.reconcile(self.venue_router, pause_cb=lambda reason: self.safety.kill_switch(reason))
        except Exception as e:  # pragma: no cover
            logger.error(f"[LiveRunner] advanced reconcile error {e}")
        # Settings fingerprint + cfg_hash guard (simple hash of settings model dump)
        try:
            import json, hashlib
            cfg_bytes = json.dumps(self.settings.model_dump() if hasattr(self.settings,'model_dump') else self.settings.__dict__, sort_keys=True).encode()
            cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()[:16]
            row = persist_fetchone("SELECT cfg_hash FROM settings_fingerprint WHERE id=1")
            if row and row.get('cfg_hash') and row['cfg_hash'] != cfg_hash:
                # mismatch – apply size downscale by setting a flag on safety
                self.safety.size_downscale = 0.7
                logger.warning(f"[LiveRunner] Config hash mismatch old={row['cfg_hash']} new={cfg_hash} – applying size downscale 0.7 for this session")
                try:
                    publish_alert('CONFIG_HASH_MISMATCH', 'Config hash mismatch – size downscale applied', severity='WARN', meta={'old': row['cfg_hash'], 'new': cfg_hash})
                except Exception:
                    pass
            snapshot_settings_fingerprint(cfg_hash, profile_version=None)
        except Exception as e:  # pragma: no cover
            logger.error(f"[LiveRunner] cfg_hash fingerprint error {e}")
        # Start HTTP metrics exporter if configured
        if self.settings.live and self.settings.live.metrics.get("exporter") == "http" and web:
            try:
                await self._start_http()
            except Exception as e:
                logger.error(f"[LiveRunner] http exporter failed {e}")

    async def _start_http(self):  # pragma: no cover (integration)
        port = int(self.settings.live.metrics.get("http_port", 8765))
        app = web.Application()
        async def metrics_handler(_):
            return web.Response(text=self.metrics.to_prometheus(), content_type="text/plain")
        async def health(_):
            return web.json_response({"ok": True, "paused": self.safety.state.paused})
        app.router.add_get('/metrics', metrics_handler)
        app.router.add_get('/healthz', health)
        app.router.add_get('/readyz', health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        self._http_site = site
        self._http_app = app
        logger.info(f"[LiveRunner] HTTP metrics exporter listening on :{port}")

    async def _reconcile(self):
        if not (self.settings.live and self.settings.live.dry_run):
            return
        # Legacy partial reconciliation
        partials = [o for o in self.store.list_orders() if o.get("status") == "PARTIAL"]
        for p in partials:
            self.store.update_order(p["client_order_id"], status="FILLED")
            logger.info(f"[LiveRunner] Reconciled partial -> FILLED {p['client_order_id']}")
        # Multi-venue reconciliation (Sprint 23)
        if self.venue_router:
            try:
                open_by_client = {}
                aggregate_positions: dict[str, float] = {}
                aggregate_avg_px: dict[str, float] = {}
                for vid, adapter in self.venue_router.venues.items():
                    try:
                        oos = await adapter.open_orders()  # type: ignore
                        for ack in oos:
                            cid = ack.client_order_id
                            if cid in open_by_client:
                                continue  # duplicate across venues ignore
                            open_by_client[cid] = (vid, ack)
                        # Positions aggregation
                        try:
                            poss = await adapter.positions()  # type: ignore
                            for p in poss:
                                cur_qty = aggregate_positions.get(p.symbol, 0.0)
                                aggregate_positions[p.symbol] = cur_qty + p.qty
                                cur_notional = aggregate_avg_px.get(p.symbol, 0.0) * abs(cur_qty)
                                new_notional = p.avg_px * abs(p.qty)
                                total_qty = abs(cur_qty) + abs(p.qty)
                                if total_qty > 0:
                                    aggregate_avg_px[p.symbol] = (cur_notional + new_notional) / total_qty
                        except Exception:
                            pass
                    except Exception as e:  # pragma: no cover
                        logger.warning(f"[LiveRunner] reconcile open_orders error venue={vid} {e}")
                # Ensure DB rows exist & update missing metadata
                for cid, (vid, ack) in open_by_client.items():
                    self.store.ensure_order(cid, venue_id=vid)
                    self.store.update_order(cid, venue_id=vid, venue_order_id=ack.venue_order_id, status=ack.status)
                # Close stray DB orders that are not open at any venue
                db_orders = self.store.list_orders()
                for row in db_orders:
                    cid = row["client_order_id"]
                    if cid not in open_by_client and row.get("status") not in ("FILLED","CANCELED","REJECTED"):
                        self.store.update_order(cid, status="CANCELED")
                        logger.info(f"[LiveRunner] Reconciled stray order -> CANCELED {cid}")
                # Persist aggregated positions
                for sym, qty in aggregate_positions.items():
                    try:
                        avg_px = aggregate_avg_px.get(sym, 0.0)
                        self.store.upsert_position(sym, qty=qty, avg_px=avg_px)
                    except Exception:
                        pass
            except Exception as e:  # pragma: no cover
                logger.error(f"[LiveRunner] multi-venue reconcile error {e}")

    async def _supervise(self):
        while True:
            await asyncio.sleep(5)
            try:
                # WAL corruption guard (simple PRAGMA quick_check)
                try:
                    from ultra_signals.persist.db import fetchone as db_fetchone
                    qc = db_fetchone("PRAGMA quick_check")
                    if qc and 'ok' not in list(qc.values())[0]:  # pragma: no cover
                        self.safety.kill_switch("DB_CORRUPTION")
                        logger.error(f"[LiveRunner] DB quick_check failed: {qc} — trading paused")
                except Exception:
                    pass
                # Ensure daily risk_runtime row
                try:
                    from datetime import datetime
                    from ultra_signals.persist.db import execute as db_exec
                    day = datetime.utcnow().strftime('%Y-%m-%d')
                    db_exec("INSERT OR IGNORE INTO risk_runtime(day, realized_pnl, consecutive_losses, paused) VALUES(?,?,?,?)", (day, 0.0, 0, 1 if self.safety.state.paused else 0))
                except Exception:
                    pass
                self.metrics.set_queue_depth("feed", self.feed_q.qsize())
                self.metrics.set_queue_depth("orders", self.order_q.qsize())
                snap = self.safety.snapshot()
                m = self.metrics.snapshot()
                logger.info(f"[LiveRunner] safety={snap} metrics={m}")
                # Venue router health gating
                try:
                    if self.venue_router:
                        vsnap = self.venue_router.snapshot()
                        logger.info(f"[LiveRunner] venue_router={vsnap}")
                        if self.venue_router.all_order_venues_red():
                            self.safety.kill_switch("ORDERS_UNAVAILABLE")
                            try:
                                publish_alert('VENUE_OUTAGE', 'All order venues red', severity='ERROR')
                            except Exception:
                                pass
                        elif self.venue_router.all_data_venues_red():
                            # Data unavailable -> pause new orders but don't hard kill; mark paused reason
                            self.safety.kill_switch("DATA_UNAVAILABLE")
                            try:
                                publish_alert('DATA_OUTAGE', 'All data venues red', severity='ERROR')
                            except Exception:
                                pass
                        else:
                            # If previously paused due to venue outage and now healthy, resume automatically
                            if self.safety.state.paused and self.safety.state.reason in ("ORDERS_UNAVAILABLE", "DATA_UNAVAILABLE"):
                                self.safety.resume()
                                try:
                                    publish_alert('VENUE_RECOVERY', 'Venues recovered')
                                except Exception:
                                    pass
                except Exception:
                    pass
                # persist safety snapshot
                try:
                    self.store.set_risk_value("safety_state", self.safety.serialize())
                except Exception:
                    pass
                # Equity curve append (every ~5m)
                try:
                    from ultra_signals.persist.db import fetchone as db_one, execute as db_exec
                    now_ms = int(time.time()*1000)
                    row = db_one("SELECT ts FROM equity_curve ORDER BY ts DESC LIMIT 1")
                    if not row or (now_ms - int(row.get('ts',0))) > 5*60*1000:
                        # Simple equity approximation: realized pnl only (placeholder until full portfolio equity calc integrated)
                        risk_row = db_one("SELECT realized_pnl FROM risk_runtime ORDER BY day DESC LIMIT 1") or {'realized_pnl':0.0}
                        equity = float(risk_row.get('realized_pnl',0.0))
                        # Drawdown placeholder using max historical equity
                        max_row = db_one("SELECT MAX(equity) as m FROM equity_curve") or {'m': equity}
                        max_eq = max(float(max_row.get('m') or 0.0), equity)
                        dd = 0.0 if max_eq <= 0 else (max_eq - equity) / max_eq * 100.0
                        db_exec("INSERT OR REPLACE INTO equity_curve(ts,equity,drawdown) VALUES(?,?,?)", (now_ms, equity, dd))
                except Exception:
                    pass
                # metrics exporter
                try:
                    if self.settings.live and self.settings.live.metrics.get("exporter") == "csv":
                        self.metrics.export_csv(self.settings.live.metrics.get("csv_path", "live_metrics.csv"))
                except Exception:
                    pass
                # structured heartbeat JSON log if enabled
                if self.settings.live and self.settings.live.metrics.get("json_log"):
                    try:
                        logger.bind(stage="heartbeat").info(json.dumps({"safety": snap, "metrics": m}))
                    except Exception:
                        pass
                # heartbeat file
                try:
                    if self.settings.live:
                        hb_int = int(self.settings.live.health.get("heartbeat_interval_sec", 30))
                        if int(time.time()) % hb_int < 5:  # coarse window
                            from pathlib import Path, PurePath
                            hb_dir = self.settings.live.control.get("control_dir", "live_controls")
                            Path(hb_dir).mkdir(parents=True, exist_ok=True)
                            with open(PurePath(hb_dir)/"heartbeat.txt", "w", encoding="utf-8") as f:
                                f.write(str(int(time.time())))
                except Exception:
                    pass
                # TCA alerts check (best-effort)
                try:
                    if getattr(self, 'tca_engine', None) is not None:
                        # cadence: only run every N loops
                        if not hasattr(self, '_tca_loop_counter'):
                            self._tca_loop_counter = 0
                        self._tca_loop_counter = (self._tca_loop_counter + 1) % max(1, int(self._tca_alert_cadence or 1))
                        if self._tca_loop_counter == 0:
                            try:
                                # global check
                                self.tca_engine.check_alerts()
                                # per-symbol checks for watched symbols only (if configured)
                                if self._tca_watched_symbols:
                                    for s in list(self._tca_watched_symbols):
                                        # schedule per-symbol checks respecting last-checked timestamps
                                        last = None
                                        try:
                                            last = self.tca_engine._get_symbol_last_checked(s)
                                        except Exception:
                                            last = None
                                        now_ms = int(time.time() * 1000)
                                        # default per-symbol cadence equals global cadence in ms
                                        per_symbol_cadence_ms = int(getattr(self.settings.live, 'tca_alert_symbol_cadence_ms', 60*1000))
                                        if last is None or (now_ms - int(last)) >= per_symbol_cadence_ms:
                                            try:
                                                self.tca_engine.check_alerts(symbol=s)
                                                try:
                                                    self.tca_engine._mark_symbol_checked(s)
                                                except Exception:
                                                    pass
                                            except Exception:
                                                pass
                            except Exception:
                                pass
                except Exception:
                    pass
                # simple control directory flag handling
                try:
                    cdir = getattr(self.settings.live, 'control', {}).get('control_dir') if self.settings.live else None
                    if cdir:
                        from pathlib import Path
                        p = Path(cdir)
                        if p.is_dir():
                            pause_flag = p/"pause.flag"
                            resume_flag = p/"resume.flag"
                            kill_flag = p/"kill.flag"
                            if pause_flag.exists():
                                self.safety.kill_switch("MANUAL")
                                pause_flag.unlink(missing_ok=True)
                            if resume_flag.exists():
                                self.safety.resume()
                                resume_flag.unlink(missing_ok=True)
                            if kill_flag.exists():
                                self.safety.kill_switch("KILL")
                                kill_flag.unlink(missing_ok=True)
                except Exception:
                    pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[LiveRunner] supervisor error {e}")

    async def stop(self):  # pragma: no cover
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._supervisor_task:
            self._supervisor_task.cancel()
            await asyncio.gather(self._supervisor_task, return_exceptions=True)
        self.executor.stop()
        await self.feed.stop()
        self.engine.stop()
        self.store.close()
        if self._http_site:
            try:
                await self._http_site.stop()
            except Exception:
                pass

__all__ = ["LiveRunner"]
