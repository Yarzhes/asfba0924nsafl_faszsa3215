from typing import List, Dict, Any
import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

from ultra_signals.backtest.event_runner import EventRunner
from ultra_signals.backtest.data_adapter import DataAdapter
from ultra_signals.backtest import metrics as backtest_metrics


def _trades_to_df(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    # ensure common columns used by compute_kpis
    if 'exit_time' not in df.columns and 'exit_ts' in df.columns:
        df = df.rename(columns={'exit_ts': 'exit_time'})
    if 'pnl' not in df.columns and 'PnL' in df.columns:
        df = df.rename(columns={'PnL': 'pnl'})
    return df


def _compute_lead_times(trades: List[Dict[str, Any]], feature_store, symbol: str, timeframe: str) -> Dict[str, Any]:
    """Compute lead-time from DC event timestamp to trade entry timestamp.

    Requires that the FeatureStore stores DC event-derived features under the key 'dc'
    in the per-timestamp feature bucket (as implemented by the DC integration hook).
    Returns dict with arrays: bars, seconds, and matched_count.
    """
    bars = []
    secs = []
    matched = 0
    for t in trades:
        # try to get entry timestamp from common keys
        entry_ts = t.get('entry_time') or t.get('entry_ts') or t.get('entry') or t.get('open_time')
        if entry_ts is None:
            continue
        try:
            ts = feature_store._to_timestamp(entry_ts)
        except Exception:
            ts = None
        if ts is None:
            continue

        # get latest features at or before entry ts
        feats = None
        try:
            feats = feature_store.get_features(symbol, timeframe, ts)
        except Exception:
            try:
                feats = feature_store.get_features(symbol, ts)
            except Exception:
                feats = None

        if not feats or 'dc' not in feats:
            continue
        dc = feats['dc']
        # Expect dc to contain 'last_dc_ts' or event timestamp in seconds/ms
        last_dc_ts = dc.get('last_dc_ts') or dc.get('dc_timestamp') or dc.get('last_event_ts')
        if last_dc_ts is None:
            continue
        try:
            dc_ts = feature_store._to_timestamp(last_dc_ts)
        except Exception:
            dc_ts = None
        if dc_ts is None:
            continue

        # compute bars difference (approx by comparing pd.Timestamp normalized to seconds)
        delta = ts - dc_ts
        if not isinstance(delta, pd.Timedelta):
            continue
        bars.append(delta // pd.Timedelta('1S'))
        secs.append(delta.total_seconds())
        matched += 1

    return {'bars': bars, 'seconds': secs, 'matched': matched}


def run_ab_backtest(settings: Dict[str, Any], symbol: str, timeframe: str, prices: List[float], out_dir: str):
    """Run two quick backtests: baseline vs DC-feature-enhanced.

    This is a convenience harness for small experiments. It requires a DataAdapter
    that can provide ohlcv for the symbol (we can create a simple CSV-backed adapter via DataAdapter).
    Outputs KPIs and lead-time plots into out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Use DataAdapter to prepare a minimal CSV-backed store if needed
    adapter = DataAdapter(settings.get('backtest', {}).get('data', {}))

    # Baseline run
    runner = EventRunner(settings, adapter, signal_engine=settings.get('signal_engine_placeholder'), feature_store=settings.get('feature_store'))
    trades_b, equity_b = runner.run(symbol, timeframe)

    # DC run
    runner_dc = EventRunner(settings, adapter, signal_engine=settings.get('signal_engine_placeholder'), feature_store=settings.get('feature_store'))
    trades_dc, equity_dc = runner_dc.run(symbol, timeframe)

    # Convert trades to DataFrame and compute KPIs
    df_b = _trades_to_df(trades_b)
    df_dc = _trades_to_df(trades_dc)
    kpis_b = backtest_metrics.compute_kpis(df_b)
    kpis_dc = backtest_metrics.compute_kpis(df_dc)

    # Save KPI summaries
    summary = {'baseline': kpis_b, 'dc': kpis_dc}
    try:
        import json
        with open(os.path.join(out_dir, 'ab_kpis.json'), 'w') as fh:
            json.dump(summary, fh, indent=2)
    except Exception:
        pass

    # Compute lead-time metrics for DC run only (how far ahead DC events signalled prior to entry)
    feature_store = settings.get('feature_store')
    lead = {'bars': [], 'seconds': [], 'matched': 0}
    if feature_store is not None:
        lead = _compute_lead_times(trades_dc, feature_store, symbol, timeframe)
        # simple histograms
        plt.figure(figsize=(6,3))
        if lead['seconds']:
            plt.hist(lead['seconds'], bins=30)
            plt.title('Lead time to entry (seconds)')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'lead_seconds_hist.png'))
            plt.close()

        plt.figure(figsize=(6,3))
        if lead['bars']:
            # bars might be pandas Timedelta seconds ints; convert
            bars_vals = [int(b) for b in lead['bars']]
            plt.hist(bars_vals, bins=30)
            plt.title('Lead time to entry (seconds bucket)')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'lead_bars_hist.png'))
            plt.close()

    # Equity curve plot (baseline vs dc)
    plt.figure(figsize=(6,3))
    if equity_b:
        xs = [i for i,_ in enumerate(equity_b)]
        ys = [r['equity'] for r in equity_b]
        plt.plot(xs, ys, label='baseline')
    if equity_dc:
        xs = [i for i,_ in enumerate(equity_dc)]
        ys = [r['equity'] for r in equity_dc]
        plt.plot(xs, ys, label='dc')
    plt.legend(); plt.title('A/B Equity'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ab_equity.png'))
    plt.close()

    return {'kpis': summary, 'lead': lead}
