#!/usr/bin/env python3
"""Simple sniper system validation"""

from ultra_signals.core.config import load_settings
from ultra_signals.engine.sniper_counters import get_sniper_counters

def test_sniper_system():
    try:
        # Load settings
        settings = load_settings('settings.yaml')
        print(f"✅ Settings loaded: sniper_mode.enabled = {settings.runtime.sniper_mode.enabled}")
        
        # Convert to dict format for sniper_counters compatibility
        settings_dict = {
            'redis': settings.redis.model_dump() if settings.redis else {},
            'runtime': {
                'sniper_mode': settings.runtime.sniper_mode.model_dump()
            }
        }
        
        # Initialize sniper counters
        counters = get_sniper_counters(settings_dict)
        print(f"✅ Sniper counters initialized: {type(counters).__name__}")
        
        # Check Redis status
        redis_status = getattr(counters, '_use_redis', False)
        print(f"✅ Redis backend: {redis_status}")
        
        # Test counter operation
        result = counters.check_and_increment(2, 6)  # 2 per hour, 6 per day
        print(f"✅ Test signal allowed: {result is None}")
        
        print("🎯 Sniper system validation complete!")
        return True
        
    except Exception as e:
        print(f"❌ Sniper system error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_sniper_system()
