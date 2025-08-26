import asyncio, time, http.client
import pytest
from ultra_signals.live.runner import LiveRunner
from ultra_signals.core.config import Settings, RuntimeSettings, LiveSettings, LiveLatencySettings, LiveLatencyBudget, LiveQueueSettings, LiveRetrySettings, LiveCircuitBreakers, LiveOrderErrorBurst
from ultra_signals.live.state_store import StateStore


def minimal_settings(tmp_path):
    return Settings(
        data_sources={"binance": {"api_key": "x", "api_secret": "y"}},
        runtime=RuntimeSettings(symbols=["BTCUSDT"], timeframes=["5m"], primary_timeframe="5m", reconnect_backoff_ms=1000),
        features=...,
        derivatives=..., regime=..., weights_profiles=..., filters=..., engine=..., ensemble=..., correlation=..., portfolio=..., brakes=..., sizing=..., funding_rate_provider=..., transport=..., live=LiveSettings(enabled=True,dry_run=True,queues=LiveQueueSettings(),retries=LiveRetrySettings(),circuit_breakers=LiveCircuitBreakers(order_error_burst=LiveOrderErrorBurst()),latency=LiveLatencySettings(),metrics={"exporter":"http","http_port": 8899, "json_log": False}, symbols=["BTCUSDT"], timeframes=["5m"])
    )  # placeholder ellipsis for required sections not used in test


@pytest.mark.asyncio
async def test_reconcile_partial(monkeypatch, tmp_path):
    # Create a runner with a pre-existing partial order
    # We bypass full Settings complexity by monkeypatching needed attrs
    class Dummy:
        pass
    settings = Dummy()
    settings.live = Dummy(); settings.live.dry_run = True; settings.live.metrics={"exporter":"none"}; settings.live.health={}; settings.live.control={}; settings.engine = Dummy(); settings.engine.risk = Dummy(); settings.engine.risk.max_spread_pct={"default":0.05}
    settings.data_sources={"binance": Dummy()}; settings.data_sources["binance"].api_key="k"; settings.data_sources["binance"].api_secret="s"
    settings.runtime = Dummy(); settings.runtime.symbols=["BTCUSDT"]; settings.runtime.timeframes=["5m"]; settings.runtime.primary_timeframe="5m"
    lr = LiveRunner(settings, dry_run=True)
    # insert partial order directly
    store = lr.store
    oid = "TESTPARTIAL"
    store.ensure_order(oid)
    store.update_order(oid, status="PARTIAL")
    await lr.start()
    await asyncio.sleep(0.1)
    rec = store.get_order(oid)
    assert rec["status"] == "FILLED"  # reconciled
    await lr.stop()


@pytest.mark.asyncio
async def test_http_metrics_exporter(monkeypatch):
    class Dummy:
        pass
    settings = Dummy()
    settings.live = Dummy(); settings.live.dry_run = True; settings.live.metrics={"exporter":"http","http_port": 8898, "json_log": False, "health":{} }; settings.live.health={}; settings.live.control={}
    settings.engine = Dummy(); settings.engine.risk = Dummy(); settings.engine.risk.max_spread_pct={"default":0.05}
    settings.data_sources={"binance": Dummy()}; settings.data_sources["binance"].api_key="k"; settings.data_sources["binance"].api_secret="s"
    settings.runtime = Dummy(); settings.runtime.symbols=["BTCUSDT"]; settings.runtime.timeframes=["5m"]; settings.runtime.primary_timeframe="5m"
    lr = LiveRunner(settings, dry_run=True)
    await lr.start()
    # fetch metrics
    await asyncio.sleep(0.1)
    async def fetch():
        def _do():
            conn = http.client.HTTPConnection("localhost", 8898, timeout=2)
            conn.request("GET", "/metrics")
            r = conn.getresponse()
            return r.read().decode()
        return await asyncio.to_thread(_do)
    data = await fetch()
    assert "orders_sent_total" in data
    await lr.stop()