"""
Sniper Mode Counters with Redis Backend

Provides distributed sniper-mode signal counting for multi-process deployments.
Falls back to in-memory counting if Redis is unavailable.
"""

import time
import json
from typing import Optional, Dict, Any
from collections import deque
from loguru import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


class SniperCounters:
    """Thread-safe, Redis-backed sniper signal counting with in-memory fallback."""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self._redis_client = None
        self._memory_fallback = {'hour': deque(), 'day': deque()}
        self._use_redis = False
        
        # Try to initialize Redis if available and configured
        if REDIS_AVAILABLE:
            redis_cfg = settings.get('redis', {})
            if redis_cfg.get('enabled', False):
                try:
                    self._redis_client = redis.Redis(
                        host=redis_cfg.get('host', 'localhost'),
                        port=redis_cfg.get('port', 6379),
                        db=redis_cfg.get('db', 0),
                        password=redis_cfg.get('password'),
                        socket_timeout=redis_cfg.get('timeout', 5),
                        decode_responses=True
                    )
                    # Test connection
                    self._redis_client.ping()
                    self._use_redis = True
                    logger.info("SniperCounters: Using Redis backend")
                except Exception as e:
                    logger.warning(f"SniperCounters: Redis unavailable, falling back to memory: {e}")
        
        if not self._use_redis:
            logger.info("SniperCounters: Using in-memory backend")
    
    def _redis_key(self, window: str) -> str:
        """Generate Redis key for the time window."""
        now = int(time.time())
        if window == 'hour':
            bucket = now // 3600
        elif window == 'day':
            bucket = now // 86400
        else:
            raise ValueError(f"Invalid window: {window}")
        return f"sniper_signals:{window}:{bucket}"
    
    def _cleanup_redis(self, window: str) -> None:
        """Clean up old Redis keys for the window type."""
        try:
            now = int(time.time())
            if window == 'hour':
                cutoff = now - 3600
                pattern = "sniper_signals:hour:*"
            elif window == 'day':
                cutoff = now - 86400
                pattern = "sniper_signals:day:*"
            else:
                return
            
            # Find and delete old keys
            for key in self._redis_client.scan_iter(match=pattern):
                try:
                    bucket_str = key.split(':')[-1]
                    bucket_time = int(bucket_str) * (3600 if window == 'hour' else 86400)
                    if bucket_time < cutoff:
                        self._redis_client.delete(key)
                except (ValueError, IndexError):
                    pass  # Skip malformed keys
        except Exception as e:
            logger.warning(f"Redis cleanup failed for {window}: {e}")
    
    def _get_redis_count(self, window: str) -> int:
        """Get current count from Redis for the window."""
        try:
            key = self._redis_key(window)
            count = self._redis_client.get(key)
            return int(count) if count else 0
        except Exception as e:
            logger.warning(f"Redis get failed for {window}: {e}")
            return 0
    
    def _increment_redis(self, window: str) -> int:
        """Increment Redis counter and return new count."""
        try:
            key = self._redis_key(window)
            # Set TTL to twice the window size for automatic cleanup
            ttl = 7200 if window == 'hour' else 172800
            
            pipe = self._redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl)
            results = pipe.execute()
            return int(results[0])
        except Exception as e:
            logger.warning(f"Redis increment failed for {window}: {e}")
            return 0
    
    def _memory_cleanup(self, window: str) -> None:
        """Clean up old timestamps from memory deque."""
        now = int(time.time())
        cutoff = now - (3600 if window == 'hour' else 86400)
        dq = self._memory_fallback[window]
        
        while dq and dq[0] < cutoff:
            dq.popleft()
    
    def check_and_increment(self, max_per_hour: int, daily_cap: int) -> Optional[str]:
        """
        Check sniper limits and increment if allowed.
        
        Returns:
            None if signal is allowed (and count incremented)
            String reason if signal should be blocked
        """
        if self._use_redis:
            return self._check_and_increment_redis(max_per_hour, daily_cap)
        else:
            return self._check_and_increment_memory(max_per_hour, daily_cap)
    
    def _check_and_increment_redis(self, max_per_hour: int, daily_cap: int) -> Optional[str]:
        """Redis-backed check and increment."""
        try:
            # Cleanup old keys periodically (1% chance)
            if time.time() % 100 < 1:
                self._cleanup_redis('hour')
                self._cleanup_redis('day')
            
            # Check current counts
            hour_count = self._get_redis_count('hour')
            day_count = self._get_redis_count('day')
            
            # Apply limits
            if max_per_hour > 0 and hour_count >= max_per_hour:
                return 'SNIPER_HOURLY_CAP'
            if daily_cap > 0 and day_count >= daily_cap:
                return 'SNIPER_DAILY_CAP'
            
            # Increment both counters
            self._increment_redis('hour')
            self._increment_redis('day')
            return None
            
        except Exception as e:
            logger.error(f"Redis sniper check failed, falling back to memory: {e}")
            return self._check_and_increment_memory(max_per_hour, daily_cap)
    
    def _check_and_increment_memory(self, max_per_hour: int, daily_cap: int) -> Optional[str]:
        """Memory-backed check and increment."""
        now = int(time.time())
        
        # Cleanup old entries
        self._memory_cleanup('hour')
        self._memory_cleanup('day')
        
        # Check limits
        if max_per_hour > 0 and len(self._memory_fallback['hour']) >= max_per_hour:
            return 'SNIPER_HOURLY_CAP'
        if daily_cap > 0 and len(self._memory_fallback['day']) >= daily_cap:
            return 'SNIPER_DAILY_CAP'
        
        # Increment
        self._memory_fallback['hour'].append(now)
        self._memory_fallback['day'].append(now)
        
        # Keep deques bounded
        max_keep = max(max_per_hour * 2 if max_per_hour > 0 else 100, 100)
        while len(self._memory_fallback['hour']) > max_keep:
            self._memory_fallback['hour'].popleft()
        while len(self._memory_fallback['day']) > max_keep * 24:
            self._memory_fallback['day'].popleft()
        
        return None
    
    def get_current_counts(self) -> Dict[str, int]:
        """Get current signal counts for observability."""
        if self._use_redis:
            return {
                'hour': self._get_redis_count('hour'),
                'day': self._get_redis_count('day')
            }
        else:
            self._memory_cleanup('hour')
            self._memory_cleanup('day')
            return {
                'hour': len(self._memory_fallback['hour']),
                'day': len(self._memory_fallback['day'])
            }
    
    def reset(self) -> None:
        """Reset all counters (useful for testing)."""
        if self._use_redis:
            try:
                # Delete all sniper keys
                for pattern in ["sniper_signals:hour:*", "sniper_signals:day:*"]:
                    for key in self._redis_client.scan_iter(match=pattern):
                        self._redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Redis reset failed: {e}")
        
        # Always reset memory fallback
        self._memory_fallback = {'hour': deque(), 'day': deque()}


# Global instance cache
_sniper_counters: Optional[SniperCounters] = None


def get_sniper_counters(settings: Dict[str, Any]) -> SniperCounters:
    """Get or create singleton SniperCounters instance."""
    global _sniper_counters
    if _sniper_counters is None:
        _sniper_counters = SniperCounters(settings)
    return _sniper_counters


def reset_sniper_counters() -> None:
    """Reset global counters (for testing)."""
    global _sniper_counters
    if _sniper_counters:
        _sniper_counters.reset()
    _sniper_counters = None
