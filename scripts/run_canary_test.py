#!/usr/bin/env python3
"""
Canary Test Runner - Single Symbol Live Validation
Runs live trading on BTCUSDT only for validation before full rollout.
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
import re
from collections import defaultdict, Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def _parse_prometheus_metrics(prom_path: Path):
    data = {
        'signals_candidates_total': 0,
        'signals_allowed_total': 0,
        'signals_blocked_total': 0,
        'blocked_reasons': Counter(),
        'latency': {},
    }
    if not prom_path.exists():
        return data
    txt = prom_path.read_text(encoding='utf-8')
    for line in txt.splitlines():
        if line.startswith('signals_candidates_total'):
            try:
                data['signals_candidates_total'] = int(float(line.split()[-1]))
            except Exception:
                pass
        elif line.startswith('signals_allowed_total'):
            try:
                data['signals_allowed_total'] = int(float(line.split()[-1]))
            except Exception:
                pass
        elif line.startswith('signals_blocked_total'):
            try:
                data['signals_blocked_total'] = int(float(line.split()[-1]))
            except Exception:
                pass
        elif line.startswith('#'):
            continue
        # dynamic block reasons exported as counters like block_reason_total
        elif line.startswith('block_') and line.endswith(tuple('0123456789')):
            parts = line.split()
            if len(parts) == 2:
                metric, val = parts
                reason = metric.replace('block_', '').replace('_total','')
                try:
                    data['blocked_reasons'][reason.upper()] = int(float(val))
                except Exception:
                    pass
        elif 'latency_tick_to_decision{quantile="0.5"}' in line:
            try:
                data['latency']['p50'] = float(line.split()[-1])
            except Exception:
                pass
        elif 'latency_tick_to_decision{quantile="0.9"}' in line:
            try:
                data['latency']['p90'] = float(line.split()[-1])
            except Exception:
                pass
    return data

def _collect_pre_samples(log_path: Path, max_samples: int = 5):
    samples = []
    if not log_path.exists():
        return samples
    pattern = re.compile(r"p:(?P<pwin>0\.\d+) \| regime:(?P<reg>[^| ]+)")
    for line in reversed(log_path.read_text(encoding='utf-8').splitlines()):
        m = pattern.search(line)
        if m:
            samples.append({'p_win': float(m.group('pwin')), 'regime': m.group('reg')})
            if len(samples) >= max_samples:
                break
    return list(reversed(samples))

def save_canary_results(start_time, end_time, duration_minutes, test_mode="CANARY"):
    """Save canary test results with signal metrics histogram and PRE samples."""
    results_path = Path("reports/canary_results.md")
    results_path.parent.mkdir(exist_ok=True)
    actual_duration = (end_time - start_time).total_seconds() / 60

    prom_path = Path('reports/canary_metrics.prom')
    prom = _parse_prometheus_metrics(prom_path)
    log_path = Path('reports/canary_run.log')
    pre_samples = _collect_pre_samples(log_path)

    # Derive top veto reasons histogram (blocked_reasons)
    reasons_md = "None"
    if prom['blocked_reasons']:
        top = prom['blocked_reasons'].most_common(10)
        reasons_md = "\n".join(f"| {r} | {c} |" for r,c in top)

    pre_md = "\n".join(f"- p_win={s['p_win']:.2f} regime={s['regime']}" for s in pre_samples) or "(none)"

    # Single-knob recommendation (heuristic)
    recommendation = "Hold thresholds"  # default
    if prom['signals_candidates_total'] == 0:
        recommendation = "Lower ensemble.confidence_floor 0.65->0.63 to admit initial candidates"
    elif prom['signals_allowed_total'] == 0 and prom['signals_blocked_total'] > 0:
        recommendation = "Relax funding_rate_limit 0.00035->0.0004 if funding rejections dominate"
    elif prom['signals_allowed_total'] > 50:
        recommendation = "Tighten VPIN / spread veto (increase wide_spread_bps or confidence_floor)"

    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Canary Results

## Window
Start: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}  
End: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}  
Planned: {duration_minutes}m  
Actual: {actual_duration:.1f}m  
Mode: {test_mode}

## Signal Metrics
- signals_candidates_total: {prom['signals_candidates_total']}
- signals_blocked_total: {prom['signals_blocked_total']}
- signals_allowed_total: {prom['signals_allowed_total']}
- latency p50: {prom['latency'].get('p50','NA')} ms  
- latency p90: {prom['latency'].get('p90','NA')} ms

## Top Veto Reasons
| Reason | Count |
|--------|-------|
{reasons_md}

## PRE Samples (latest)
{pre_md}

## Recommendation
{recommendation}

*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*
""")

    print(f"\n📄 Canary results saved to: {results_path}")
    return results_path

def timeout_handler(duration_minutes, start_time):
    """Handle test timeout and graceful shutdown"""
    time.sleep(duration_minutes * 60)
    end_time = datetime.now(timezone.utc)
    print(f"\n⏰ {duration_minutes} minutes elapsed - stopping canary test...")
    
    # Save results
    save_canary_results(start_time, end_time, duration_minutes)
    print(f"✅ Canary test results saved: canary_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md")
    print("🛑 Canary test completed - duration limit reached")
    
    # Force exit
    os._exit(0)

def main():
    # Manual arg parse (robust against unknown extra args)
    raw = sys.argv[1:]
    duration = 1440
    config = 'settings_canary.yaml'
    i = 0
    while i < len(raw):
        a = raw[i]
        if a.startswith('--duration='):
            try: duration = int(a.split('=',1)[1])
            except: pass
        elif a == '--duration' and i+1 < len(raw):
            try: duration = int(raw[i+1]); i += 1
            except: pass
        elif a.startswith('--config='):
            config = a.split('=',1)[1]
        elif a == '--config' and i+1 < len(raw):
            config = raw[i+1]; i += 1
        i += 1

    print(f"Canary test will run for {duration} minutes ({duration/60:.1f} hours) using config {config}")

    os.environ['TRADING_MODE'] = 'CANARY'
    os.environ['CANARY_ALL_SYMBOLS'] = 'true'
    os.environ['SETTINGS_FILE'] = config

    start_time = datetime.now(timezone.utc)
    print(f"""
🎯 Starting Canary Test - All 20 Symbols Live Validation
========================================================
• Mode: CANARY (live orders on all 20 symbols)
• Symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT
           DOGEUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, TONUSDT
           TRXUSDT, DOTUSDT, NEARUSDT, ATOMUSDT, LTCUSDT
           BCHUSDT, ARBUSDT, APTUSDT, MATICUSDT, SUIUSDT
• Timeframes: 1m, 3m, 5m, 15m
• Sniper caps: 2/hour, 6/day
• MTF confirmation: REQUIRED
• Duration: {duration} minutes ({duration/60:.1f} hours)
========================================================
⏱️  Test will auto-stop after {duration} minutes...
""")

    timer = threading.Timer(duration * 60, timeout_handler, args=[duration, start_time])
    timer.start()

    try:
        # Sanitize argv so realtime_runner only sees supported arguments
        sys.argv = [sys.argv[0], '--config', config]
        from ultra_signals.apps.realtime_runner import run as run_realtime
        run_realtime()
    except KeyboardInterrupt:
        print("\n🛑 Canary test stopped by user")
        timer.cancel()
        end_time = datetime.now(timezone.utc)
        save_canary_results(start_time, end_time, duration)
    except Exception as e:
        print(f"\n❌ Error during canary test: {e}")
        timer.cancel()
        end_time = datetime.now(timezone.utc)
        save_canary_results(start_time, end_time, duration, "CANARY_ERROR")
        raise

if __name__ == "__main__":
    main()
