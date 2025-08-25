"""High level live runner coordinating feed -> engine -> executor."""
from __future__ import annotations
import asyncio
import os
import time
from loguru import logger
from .feed_ws import FeedAdapter
from .engine_worker import EngineWorker
from .order_exec import OrderExecutor
from .state_store import StateStore
from .safety import SafetyManager
from .metrics import Metrics


class LiveRunner:
    def __init__(self, settings, dry_run: bool = True):
        self.settings = settings
        live_cfg = (getattr(settings, "live", None) or {})
        rt_cfg = getattr(settings, "runtime", None)
        latency = ((getattr(live_cfg, "latency") or {}) if hasattr(live_cfg, "latency") else {})
        qcfg = ((getattr(live_cfg, "queues") or {}) if hasattr(live_cfg, "queues") else {})
        self.feed_q = asyncio.Queue(maxsize=int(qcfg.get("feed", 2000)))
        self.engine_q = asyncio.Queue(maxsize=int(qcfg.get("engine", 256)))
        self.order_q = asyncio.Queue(maxsize=int(qcfg.get("orders", 128)))
        self.store = StateStore()
        cb_cfg = getattr(live_cfg, "circuit_breakers", None) or {}
        self.safety = SafetyManager(
            daily_loss_limit_pct=float(cb_cfg.get("daily_loss_limit_pct", 0.06)),
            max_consecutive_losses=int(cb_cfg.get("max_consecutive_losses", 4)),
            order_error_burst_count=int((cb_cfg.get("order_error_burst") or {}).get("count", 6)),
            order_error_burst_window_sec=int((cb_cfg.get("order_error_burst") or {}).get("window_sec", 120)),
            data_staleness_ms=int(cb_cfg.get("data_staleness_ms", 2500)),
        )
        self.metrics = Metrics()
        self.feed = FeedAdapter(settings, self.feed_q)
        self.engine = EngineWorker(self.feed_q, self.order_q, latency_budget_ms=int((((getattr(live_cfg,'latency') or {}) ).get('tick_to_decision_ms', {}) or {}).get('p99', 180)), metrics=self.metrics, safety=self.safety)
        self.executor = OrderExecutor(self.order_q, self.store, rate_limits=getattr(live_cfg, "rate_limits", {}) if hasattr(live_cfg, "rate_limits") else {}, retry_cfg=getattr(live_cfg, "retries", {}) if hasattr(live_cfg, "retries") else {}, dry_run=dry_run, safety=self.safety, metrics=self.metrics)
        self._tasks = []
        self._supervisor_task: asyncio.Task | None = None
        # Restore safety state (if persisted)
        try:
            saved = self.store.get_risk_value("safety_state")
            if saved:
                self.safety.restore(saved)
                logger.info("[LiveRunner] Restored safety state {}", saved)
        except Exception:
            pass

    def _check_env(self):
        if not self.settings.live.dry_run:
            src = self.settings.data_sources.get("binance", None)
            if not src or not src.api_key or not src.api_secret:
                raise RuntimeError("API keys missing for live mode")

    async def start(self):
        self._check_env()
        logger.info("[LiveRunner] Starting live pipeline (dry_run={})", self.settings.live.dry_run)
        self._tasks.append(asyncio.create_task(self.feed.run(), name="feed"))
        self._tasks.append(asyncio.create_task(self.engine.run(), name="engine"))
        self._tasks.append(asyncio.create_task(self.executor.run(), name="executor"))
        # supervise
        self._supervisor_task = asyncio.create_task(self._supervise(), name="supervisor")

    async def _supervise(self):
        while True:
            await asyncio.sleep(5)
            try:
                self.metrics.set_queue_depth("feed", self.feed_q.qsize())
                self.metrics.set_queue_depth("orders", self.order_q.qsize())
                snap = self.safety.snapshot()
                m = self.metrics.snapshot()
                logger.info(f"[LiveRunner] safety={snap} metrics={m}")
                # persist safety snapshot
                try:
                    self.store.set_risk_value("safety_state", self.safety.serialize())
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

__all__ = ["LiveRunner"]
