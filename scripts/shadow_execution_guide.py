#!/usr/bin/env python3
"""
Shadow Mode Execution Guide
Simple script to demonstrate how to run the shadow test
"""

import os
import time
import datetime

def print_banner():
    print("🎯 SNIPER MODE SHADOW TEST - EXECUTION GUIDE")
    print("=" * 60)
    print(f"📅 Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📦 Version: v0.9.3-shadow-ready")
    print(f"🔧 Mode: SHADOW (no live orders)")
    print("=" * 60)

def show_terminal_commands():
    print("\\n🚀 EXECUTION COMMANDS")
    print("-" * 30)
    print("\\n📟 Terminal A (Shadow Test Runner):")
    print("```bash")
    print("python scripts/run_shadow_test.py --duration 120")
    print("```")
    
    print("\\n📊 Terminal B (Real-time Monitor):")
    print("```bash") 
    print("python scripts/monitor_shadow.py --duration 120 --interval 30")
    print("```")

def show_expected_behavior():
    print("\\n📋 EXPECTED BEHAVIOR (120 minutes)")
    print("-" * 40)
    print("✅ Signal Throttling:")
    print("  • Max 2-4 allowed signals across BTC/ETH/SOL")
    print("  • Hourly cap (2) and daily cap (6) enforced")
    print("  • Visible rejections in Prometheus metrics")
    
    print("\\n✅ MTF Confirmation:")
    print("  • Signals blocked when timeframes disagree")
    print("  • sniper_rejections_total{reason='mtf_required'} incrementing")
    
    print("\\n✅ Telegram Integration:")
    print("  • PRE-TRADE messages only for allowed signals")
    print("  • NO trade cards sent for blocked signals")
    print("  • Message format: 'SHADOW | {pair} | {side} | ...'")
    
    print("\\n✅ System Stability:")
    print("  • Zero crashes or hangs")
    print("  • Latency P95 < 500ms")
    print("  • Memory growth < 100MB over 2 hours")
    print("  • Redis fallback to memory if server unavailable")

def show_monitoring_metrics():
    print("\\n📊 KEY METRICS TO WATCH")
    print("-" * 30)
    print("🎯 Sniper Enforcement:")
    print("  • sniper_rejections_total{reason='hourly_cap'}")
    print("  • sniper_rejections_total{reason='daily_cap'}")
    print("  • sniper_rejections_total{reason='mtf_required'}")
    
    print("\\n📈 Signal Pipeline:")
    print("  • signals_candidates_total")
    print("  • signals_blocked_total{reason='SNIPER'}")
    print("  • signals_allowed_total")
    
    print("\\n⚡ Performance:")
    print("  • latency_tick_to_decision_ms (P50, P95)")
    print("  • memory_usage_mb")
    print("  • cpu_utilization_percent")

def show_success_criteria():
    print("\\n🎯 SUCCESS CRITERIA")
    print("-" * 20)
    criteria = [
        "Caps enforced (2/hour, 6/day) and visible in metrics",
        "MTF disagreements block signals as expected", 
        "No system crashes for 120+ minutes",
        "Latency P95 consistently < 500ms",
        "Prometheus metrics updating every 30 seconds",
        "All 3 symbols (BTC/ETH/SOL) showing fresh data",
        "Redis connection stable or graceful memory fallback"
    ]
    
    for i, criteria_item in enumerate(criteria, 1):
        print(f"  {i}. ✅ {criteria_item}")

def show_next_steps():
    print("\\n🚀 AFTER SHADOW COMPLETION")
    print("-" * 30)
    print("📝 Generate Results:")
    print("  • Review logs in reports/shadow_test_*.log")
    print("  • Fill out reports/shadow_results.md template")
    print("  • Export Prometheus metrics to JSON")
    print("  • Take Grafana dashboard screenshot")
    
    print("\\n🎯 Decision Matrix:")
    print("  ✅ IF ALL CRITERIA PASS → Proceed to Canary Mode")
    print("     • python scripts/run_canary_test.py (to be created)")
    print("     • BTCUSDT only, 1/hour, 2/day caps")
    print("     • 60-minute live trading test")
    
    print("\\n  ❌ IF ANY CRITERIA FAIL → Debug & Fix")
    print("     • Review failure details in logs")
    print("     • Apply fixes and re-run shadow test")
    print("     • Consider rollback: git checkout v0.9.3-shadow-ready")

def main():
    print_banner()
    show_terminal_commands()
    show_expected_behavior()
    show_monitoring_metrics()
    show_success_criteria()
    show_next_steps()
    
    print("\\n" + "=" * 60)
    print("🟢 READY TO START SHADOW MODE TESTING")
    print("=" * 60)
    print("\\n⚠️  IMPORTANT REMINDERS:")
    print("  • Ensure API credentials are set in environment")
    print("  • Redis server optional (memory fallback available)")
    print("  • Telegram bot token configured for notifications")
    print("  • Grafana dashboard imported for monitoring")
    
    print("\\n🏁 Execute the terminal commands above to begin!")

if __name__ == "__main__":
    main()
