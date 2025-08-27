"""CLI entrypoint for events subsystem.

Examples:
  python -m ultra_signals.events.cli fetch --days 7
  python -m ultra_signals.events.cli list --from 2025-08-25 --to 2025-09-01 --symbol BTCUSDT
  python -m ultra_signals.events.cli simulate --symbol BTCUSDT --tf 5m --from 2023-01-01 --to 2023-02-01

Currently fetch uses stub providers (no-op). Listing works if user manually
populates events table or future provider implementation does.
"""
from __future__ import annotations
import argparse
import sys
import json
from datetime import datetime, timezone, timedelta
from loguru import logger
from ultra_signals.core.config import load_settings
from ultra_signals.persist.db import init_db
from ultra_signals.persist.migrations import apply_migrations
from .providers.econ_calendar import EconCalendarProvider
from .providers.crypto_incidents import CryptoIncidentsProvider
from . import store, classifier


def _dt_parse(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def cmd_fetch(args):
    settings = load_settings(args.config)
    cfg = (settings.model_dump() if hasattr(settings,'model_dump') else settings).get('event_risk') or {}
    days = int(args.days)
    now = datetime.now(tz=timezone.utc)
    to = now + timedelta(days=days)
    from_ts = int(now.timestamp() * 1000)
    to_ts = int(to.timestamp() * 1000)
    providers = []
    p_cfg = (cfg.get('providers') or {})
    if (p_cfg.get('econ_calendar') or {}).get('enabled', True):
        providers.append(EconCalendarProvider(p_cfg.get('econ_calendar')))
    if (p_cfg.get('crypto_incidents') or {}).get('enabled', True):
        providers.append(CryptoIncidentsProvider(p_cfg.get('crypto_incidents')))

    raw_events = []
    for p in providers:
        try:
            evs = p.fetch_upcoming(from_ts, to_ts)
            raw_events.extend(evs)
        except Exception as e:  # pragma: no cover
            logger.warning(
                "[event_provider_failure] provider={} err_type={} err_msg={} fallback=skip range_days={}",
                getattr(p,'provider_name','?'), type(e).__name__, str(e), days
            )

    normed = [classifier.classify(r) for r in raw_events]
    store.upsert_events(normed)
    print(f"Fetched {len(raw_events)} raw -> {len(normed)} stored")


def cmd_list(args):
    init_db()
    apply_migrations()
    start = _dt_parse(args.from_)
    end = _dt_parse(args.to)
    rows = store.load_events_window(int(start.timestamp()*1000), int(end.timestamp()*1000))
    sym = args.symbol
    if sym:
        rows = [r for r in rows if r.get('symbol_scope') == 'GLOBAL' or sym in (r.get('symbol_scope') or '').split(',')]
    for r in rows:
        st = datetime.fromtimestamp(r['start_ts']/1000, tz=timezone.utc)
        et = datetime.fromtimestamp(r['end_ts']/1000, tz=timezone.utc)
        print(f"{r['id']} | {r['category']} ({r['importance']}) | {st.isoformat()} → {et.isoformat()} | scope={r['symbol_scope']}")
    if not rows:
        print("No events in range")


def cmd_simulate(args):
    """Compare backtest slice with vs without event gating.

    Provides: PnL delta, veto/dampen bars %, dampen trade count and abstain pct.
    """
    from ultra_signals.backtest.event_runner import EventRunner
    from ultra_signals.engine.real_engine import RealSignalEngine
    from ultra_signals.core.feature_store import FeatureStore
    from ultra_signals.backtest.data_adapter import DataAdapter
    from ultra_signals.events import gate_stats, reset_caches

    settings_obj = load_settings(args.config)
    settings = settings_obj.model_dump() if hasattr(settings_obj,'model_dump') else settings_obj
    # inject date bounds into backtest config for adapter
    settings.setdefault('backtest',{})['start_date'] = args.from_
    settings.setdefault('backtest',{})['end_date'] = args.to
    symbol = args.symbol; tf = args.tf

    # Prefetch upcoming events horizon covering range
    span_days = max(1, (datetime.fromisoformat(args.to) - datetime.fromisoformat(args.from_)).days + 1)
    cmd_fetch(argparse.Namespace(config=args.config, days=span_days))

    # RUN WITH GATING
    reset_caches()
    fs = FeatureStore(); engine = RealSignalEngine(settings, fs)
    da = DataAdapter(settings)
    runner_on = EventRunner(settings, da, engine, fs)
    runner_on.run(symbol, tf)
    pnl_on = sum(t.get('pnl',0) for t in runner_on.trades)
    stats_on = gate_stats()
    em_on = getattr(runner_on,'event_metrics',{})

    # RUN WITHOUT GATING
    settings_off = json.loads(json.dumps(settings)); settings_off.setdefault('event_risk',{})['enabled'] = False
    reset_caches()
    fs2 = FeatureStore(); engine2 = RealSignalEngine(settings_off, fs2)
    da2 = DataAdapter(settings_off)
    runner_off = EventRunner(settings_off, da2, engine2, fs2)
    runner_off.run(symbol, tf)
    pnl_off = sum(t.get('pnl',0) for t in runner_off.trades)

    result = {
        'symbol': symbol,
        'tf': tf,
        'start': args.from_,
        'end': args.to,
        'pnl_with_gating': pnl_on,
        'pnl_without_gating': pnl_off,
        'delta': pnl_on - pnl_off,
        'event_abstain_pct': stats_on.get('abstain_pct'),
        'event_veto_bars': em_on.get('veto_bars'),
        'event_dampen_bars': em_on.get('dampen_bars'),
        'event_dampen_trades': em_on.get('dampen_trades'),
        'force_closes': em_on.get('force_closes'),
        'cooldown_blocks': em_on.get('cooldown_blocks'),
        'bars': em_on.get('bars'),
    }
    print(json.dumps(result, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser("eventsctl")
    sp = p.add_subparsers(dest='cmd')
    pf = sp.add_parser('fetch')
    pf.add_argument('--config', default='settings.yaml')
    pf.add_argument('--days', default=7)
    pf.set_defaults(func=cmd_fetch)
    pl = sp.add_parser('list')
    pl.add_argument('--from', dest='from_', required=True)
    pl.add_argument('--to', dest='to', required=True)
    pl.add_argument('--symbol', default=None)
    pl.set_defaults(func=cmd_list)
    ps = sp.add_parser('simulate')
    ps.add_argument('--symbol', required=True)
    ps.add_argument('--tf', required=True)
    ps.add_argument('--from', dest='from_', required=True)
    ps.add_argument('--to', dest='to', required=True)
    ps.set_defaults(func=cmd_simulate)
    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help(); return 1
    init_db(); apply_migrations()
    return a.func(a)


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
