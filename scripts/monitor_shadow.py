#!/usr/bin/env python3
"""
Shadow Mode Monitor - Real-time Sniper Metrics Dashboard

Monitors Prometheus metrics and Redis counters during shadow test.
"""
import time
import requests
import redis
from datetime import datetime, timezone
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from ultra_signals.engine.sniper_counters import get_sniper_counters
except ImportError:
    print("Warning: Could not import sniper_counters - Redis monitoring disabled")
    get_sniper_counters = None

class ShadowMonitor:
    def __init__(self, prometheus_port=8000, redis_host='localhost', redis_port=6379):
        self.prom_url = f"http://localhost:{prometheus_port}/metrics"
        self.redis_available = False
        
        # Try to connect to Redis
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
            self.redis_client.ping()
            self.redis_available = True
            print("✅ Redis connection established")
        except Exception as e:
            print(f"⚠️ Redis not available: {e}")
            self.redis_client = None
    
    def get_prometheus_metrics(self):
        """Fetch current Prometheus metrics"""
        try:
            response = requests.get(self.prom_url, timeout=5)
            if response.status_code == 200:
                metrics = {}
                for line in response.text.split('\\n'):
                    if line.startswith('sniper_') or line.startswith('signals_'):
                        if not line.startswith('#') and '{' in line:
                            metric_name = line.split('{')[0]
                            value = line.split()[-1]
                            try:
                                metrics[metric_name] = float(value)
                            except ValueError:
                                pass
                return metrics
            return {}
        except Exception as e:
            print(f"❌ Prometheus error: {e}")
            return {}
    
    def get_redis_counters(self):
        """Fetch current Redis sniper counters"""
        if not self.redis_available or not get_sniper_counters:
            return {}
        
        try:
            counters = get_sniper_counters()
            return {
                'hourly_signals': len(counters._hourly_signals),
                'daily_signals': len(counters._daily_signals),
                'redis_keys': len(self.redis_client.keys('sniper:*')) if self.redis_client else 0
            }
        except Exception as e:
            print(f"❌ Redis counter error: {e}")
            return {}
    
    def print_status(self):
        """Print current monitoring status"""
        now = datetime.now(timezone.utc)
        print(f"\\n📊 Shadow Mode Status - {now.strftime('%H:%M:%S UTC')}")
        print("=" * 50)
        
        # Prometheus metrics
        prom_metrics = self.get_prometheus_metrics()
        if prom_metrics:
            print("🎯 Sniper Metrics:")
            for metric, value in prom_metrics.items():
                if 'sniper' in metric:
                    print(f"  {metric}: {value}")
            
            print("\\n📈 Signal Metrics:")
            for metric, value in prom_metrics.items():
                if 'signals' in metric:
                    print(f"  {metric}: {value}")
        else:
            print("⚠️ No Prometheus metrics available")
        
        # Redis counters
        redis_data = self.get_redis_counters()
        if redis_data:
            print("\\n🔄 Redis Counters:")
            for key, value in redis_data.items():
                print(f"  {key}: {value}")
        
        print("=" * 50)
    
    def monitor(self, duration_minutes=120, interval_seconds=30):
        """Run monitoring loop"""
        print(f"🔍 Starting shadow mode monitoring for {duration_minutes} minutes")
        print(f"📊 Checking metrics every {interval_seconds} seconds")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        try:
            while time.time() < end_time:
                self.print_status()
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\\n🛑 Monitoring stopped by user")
        
        print("\\n✅ Shadow mode monitoring complete")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor shadow mode test')
    parser.add_argument('--duration', type=int, default=120, 
                       help='Monitoring duration in minutes')
    parser.add_argument('--interval', type=int, default=30,
                       help='Check interval in seconds')
    parser.add_argument('--prom-port', type=int, default=8000,
                       help='Prometheus port')
    
    args = parser.parse_args()
    
    monitor = ShadowMonitor(prometheus_port=args.prom_port)
    monitor.monitor(duration_minutes=args.duration, interval_seconds=args.interval)

if __name__ == "__main__":
    main()
