#!/usr/bin/env python3
"""
Shadow Mode Test Runner - No Live Orders

This script runs the realtime_runner in shadow mode for testing sniper enforcement.
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultra_signals.apps.realtime_runner import main

def run_shadow_test():
    """Run shadow mode test with sniper enforcement"""
    print("🎯 Starting Shadow Mode Test - Sniper Enforcement Validation")
    print("=" * 60)
    print("• Mode: SHADOW (no live orders)")
    print("• Symbols: BTCUSDT, ETHUSDT, SOLUSDT")
    print("• Timeframes: 1m, 3m, 5m, 15m")
    print("• Sniper caps: 2/hour, 6/day")
    print("• MTF confirmation: REQUIRED")
    print("• Duration: 90-120 minutes")
    print("=" * 60)
    
    # Set environment for shadow test
    os.environ['ULTRA_SIGNALS_CONFIG'] = 'configs/profiles/shadow_test.yaml'
    
    # Run with shadow config
    sys.argv = ['realtime_runner', '--config', 'configs/profiles/shadow_test.yaml']
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Shadow test stopped by user")
    except Exception as e:
        print(f"❌ Shadow test error: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run shadow mode test')
    parser.add_argument('--duration', type=int, default=120, 
                       help='Test duration in minutes (default: 120)')
    
    args = parser.parse_args()
    
    print(f"Shadow test will run for {args.duration} minutes")
    run_shadow_test()
