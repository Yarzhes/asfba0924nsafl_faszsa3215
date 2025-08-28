"""Simple dry-run simulation that starts LiveRunner and simulates fills/rejects.

This script is intentionally minimal and uses the LiveRunner in dry-run mode.
It injects synthetic fill events into the TCA engine and writes some FeatureView
records to validate persistence.

Run from project root:
    python .\scripts\simulate_live_dryrun.py
"""

import asyncio
import time
import tempfile
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultra_signals.live.runner import LiveRunner
from types import SimpleNamespace


async def run_sim():
    # minimal settings object
    settings = SimpleNamespace()
    settings.live = SimpleNamespace()
    settings.live.dry_run = True
    settings.live.metrics = {}
    settings.live.tca_alert_cadence = 1
    settings.live.tca_watched_symbols = ['SYM']
    settings.live.tca_alert_symbol_cadence_ms = 1000

    # minimal engine settings needed elsewhere
    settings.engine = SimpleNamespace()
    settings.engine.risk = SimpleNamespace()
    settings.engine.risk.max_spread_pct = {'default': 0.05}

    # venues config empty
    settings.venues = {}

    runner = LiveRunner(settings, dry_run=True)

    # start components (feed run is skipped because _skip_feed heuristic)
    await runner.start()

    # simulate some fills and rejects
    tca = runner.tca_engine
    if tca is None:
        print('No TCA engine available in runner')
    else:
        # simulate a few fills across venues
        for i in range(3):
            evt = {
                'venue': 'V1',
                'symbol': 'SYM',
                'arrival_px': 100.0,
                'fill_px': 100.02 + i * 0.01,
                'filled_qty': 1.0,
                'requested_qty': 1.0,
                'arrival_ts_ms': int(time.time() * 1000),
                'completion_ts_ms': int(time.time() * 1000) + 50,
            }
            tca.record_fill(evt)
            await asyncio.sleep(0.1)
        # simulate a reject
        tca.record_reject('V2')

        # write a feature view record
        if runner.feature_writer:
            runner.feature_writer.write_record({'symbol':'SYM','ts':int(time.time()*1000),'expected_cost_bps':1.0,'realized_cost_bps':2.0,'tca_slip_bps':3.0,'components':{'note':'sim'}})
            recs = runner.feature_writer.query_recent(5)
            print('FeatureView recent:', recs)

    # let supervisor run a bit so alerts may process
    await asyncio.sleep(2)

    # stop runner
    await runner.stop()


if __name__ == '__main__':
    asyncio.run(run_sim())
