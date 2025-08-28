#!/usr/bin/env python3
"""
Sniper Mode Go-Live Pre-Flight Checklist
Run this before shadow mode to verify all systems ready
"""

import subprocess
import sys
import time
import json
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ {description}: PASSED")
            return True
        else:
            print(f"❌ {description}: FAILED")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description}: TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description}: ERROR - {e}")
        return False

def check_file_exists(filepath, description):
    """Check if required file exists"""
    print(f"🔍 {description}...")
    if Path(filepath).exists():
        print(f"✅ {description}: FOUND")
        return True
    else:
        print(f"❌ {description}: MISSING - {filepath}")
        return False

def main():
    print("🎯 SNIPER MODE GO-LIVE PRE-FLIGHT CHECKLIST")
    print("=" * 60)
    
    checks = []
    
    # 1. Core System Tests
    print("\\n📋 CORE SYSTEM VALIDATION")
    checks.append(run_command(
        "python -m pytest ultra_signals/tests/test_sniper_mode.py ultra_signals/tests/test_sniper_redis.py ultra_signals/tests/test_sniper_integration.py -v",
        "Sniper test suite"
    ))
    
    # 2. Configuration Validation
    print("\\n📋 CONFIGURATION VALIDATION")
    config_test = "from ultra_signals.core.config import load_settings; s=load_settings('settings.yaml'); print(f'Sniper enabled: {s.runtime.sniper_mode.enabled}');"
    checks.append(run_command(
        f'python -c "{config_test}"',
        "Settings loading and sniper config"
    ))
    
    # 3. Required Files
    print("\\n📋 REQUIRED FILES CHECK")
    required_files = [
        ("settings.yaml", "Main configuration file"),
        ("ultra_signals/engine/sniper_counters.py", "Sniper counter system"),
        ("ultra_signals/engine/risk_filters.py", "Risk filter integration"),
        ("dashboards/sniper-mode-dashboard.json", "Grafana dashboard"),
        ("reports/go_live_checklist.md", "Go-live checklist"),
        ("reports/readiness_report.md", "Readiness report"),
    ]
    
    for filepath, description in required_files:
        checks.append(check_file_exists(filepath, description))
    
    # 4. Git Status Check
    print("\\n📋 VERSION CONTROL STATUS")
    checks.append(run_command(
        "git log -1 --oneline",
        "Latest commit verification"
    ))
    
    checks.append(run_command(
        "git tag --list v0.9.2-sniper-safe",
        "Safety checkpoint tag exists"
    ))
    
    # 5. Quick Integration Test
    print("\\n📋 INTEGRATION TEST")
    integration_test = '''
from ultra_signals.core.config import load_settings
from ultra_signals.engine.sniper_counters import get_sniper_counters
settings = load_settings("settings.yaml")
settings_dict = {"redis": settings.redis.model_dump() if settings.redis else {}, "runtime": {"sniper_mode": settings.runtime.sniper_mode.model_dump()}}
counters = get_sniper_counters(settings_dict)
result = counters.check_and_increment(2, 6)
print("✅ Integration test passed" if result is None else "❌ Integration test failed")
'''
    
    checks.append(run_command(
        f'python -c "{integration_test}"',
        "End-to-end integration test"
    ))
    
    # Summary
    print("\\n" + "=" * 60)
    print("📊 PRE-FLIGHT CHECKLIST SUMMARY")
    print("=" * 60)
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print("\\n🟢 ALL SYSTEMS GO! Ready for shadow mode testing.")
        print("\\n🚀 Next steps:")
        print("   1. python scripts/run_shadow_test.py --duration 120")
        print("   2. python scripts/monitor_shadow.py --duration 120")
        print("   3. Review metrics in Grafana dashboard")
        print("   4. Proceed to canary mode if shadow passes")
        return 0
    else:
        print(f"\\n🔴 {total - passed} CHECKS FAILED! Address issues before proceeding.")
        print("\\n⚠️  Do not proceed to shadow mode until all checks pass.")
        return 1

if __name__ == "__main__":
    exit(main())
