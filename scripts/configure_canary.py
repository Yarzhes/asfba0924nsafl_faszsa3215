#!/usr/bin/env python3
"""
Configure Canary Test - Update settings for BTCUSDT-only live trading
"""

import yaml
import shutil
from pathlib import Path
from datetime import datetime

def backup_current_config():
    """Backup current settings before canary"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"settings_backup_{timestamp}.yaml"
    shutil.copy("settings.yaml", backup_path)
    print(f"✅ Current config backed up to: {backup_path}")
    return backup_path

def configure_canary_mode():
    """Configure settings.yaml for canary testing"""
    
    # Load current settings
    with open("settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
    
    # Canary-specific modifications
    canary_config = {
        # All 20 symbols for comprehensive canary test
        "runtime": {
            **settings.get("runtime", {}),
            "symbols": ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","TONUSDT","TRXUSDT","DOTUSDT","NEARUSDT","ATOMUSDT","LTCUSDT","BCHUSDT","ARBUSDT","APTUSDT","MATICUSDT","SUIUSDT"],  # All 20 symbols
            "sniper_mode": {
                **settings.get("runtime", {}).get("sniper_mode", {}),
                "enabled": True,
                "mtf_confirm": True,
                "min_confidence": 0.60,
                "max_signals_per_hour": 2,
                "daily_signal_cap": 6,
                "cooldown_bars": 10
            }
        },
        
        # Enable live trading (disable dry_run)
        "transport": {
            **settings.get("transport", {}),
            "telegram": {
                **settings.get("transport", {}).get("telegram", {}),
                "enabled": True
            },
            "dry_run": False  # LIVE TRADING ENABLED
        },
        
        # Portfolio settings for all 20 symbols
        "portfolio": {
            **settings.get("portfolio", {}),
            "max_exposure_per_symbol": 1000.0,  # Full exposure per symbol
            "max_total_positions": 10,          # Multiple positions allowed
            "max_positions_per_symbol": 3       # Up to 3 positions per symbol
        },
        
        # Conservative position sizing
        "sizer": {
            **settings.get("sizer", {}),
            "enabled": True,
            "base_risk_pct": 0.5,  # 0.5% risk per trade
            "max_risk_pct": 0.75,  # Max 0.75% risk
            "min_risk_pct": 0.25   # Min 0.25% risk
        },
        
        # Enhanced monitoring
        "prometheus": {
            **settings.get("prometheus", {}),
            "enabled": True,
            "port": 8000
        }
    }
    
    # Update settings with canary configuration
    for key, value in canary_config.items():
        if isinstance(value, dict) and key in settings:
            settings[key].update(value)
        else:
            settings[key] = value
    
    # Save canary configuration
    canary_settings_path = "settings_canary.yaml"
    with open(canary_settings_path, "w") as f:
        yaml.dump(settings, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Canary configuration saved to: {canary_settings_path}")
    return canary_settings_path

def main():
    print("🎯 Configuring Canary Test Setup")
    print("=" * 40)
    
    # Step 1: Backup current config
    backup_path = backup_current_config()
    
    # Step 2: Create canary configuration
    canary_path = configure_canary_mode()
    
    # Step 3: Copy canary config to main settings
    shutil.copy(canary_path, "settings.yaml")
    
    print("\n🚀 Canary Configuration Complete!")
    print("=" * 40)
    print("✅ All 20 symbols live trading enabled")
    print("✅ Production risk settings applied")
    print("✅ Sniper caps: 2/hour, 6/day")
    print("✅ Telegram notifications enabled")
    print("✅ Dry run DISABLED (live trading)")
    
    print(f"\n📁 Files created:")
    print(f"   • {backup_path} (original backup)")
    print(f"   • {canary_path} (canary template)")
    print(f"   • settings.yaml (updated for canary)")
    
    print(f"\n🎯 Ready for canary test:")
    print(f"   # 24-hour canary test (recommended)")
    print(f"   python scripts/run_canary_test.py --duration 1440")
    print(f"   ")
    print(f"   # Or 48-hour comprehensive test")
    print(f"   python scripts/run_canary_test.py --duration 2880")
    
    print(f"\n⚠️  To restore original settings after canary:")
    print(f"   cp {backup_path} settings.yaml")

if __name__ == "__main__":
    main()
