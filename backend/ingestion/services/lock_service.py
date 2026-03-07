"""
Lock Service

Redis-based distributed locking for ingestion coordination
"""
import time
import uuid
from typing import Optional


class LockService:
    """
    Distributed lock service using Redis
    
    Prevents overlapping ingestion runs
    """
    
    def __init__(self, redis_client):
        """
        Initialize lock service
        
        Args:
            redis_client: Redis client
        """
        self.redis = redis_client
        self.lock_key = "ingestion:lock"
        self.lock_ttl = 3600  # 1 hour
    
    def acquire_lock(self, timeout: int = 10) -> Optional[str]:
        """
        Acquire ingestion lock
        
        Args:
            timeout: How long to wait for lock (seconds)
            
        Returns:
            Lock token if acquired, None if failed
        """
        lock_token = str(uuid.uuid4())
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            # Try to set lock with NX (only if not exists)
            acquired = self.redis.set(
                self.lock_key,
                lock_token,
                ex=self.lock_ttl,
                nx=True
            )
            
            if acquired:
                return lock_token
            
            # Wait before retry
            time.sleep(0.5)
        
        return None
    
    def release_lock(self, lock_token: str) -> bool:
        """
        Release ingestion lock
        
        Args:
            lock_token: Token from acquire_lock
            
        Returns:
            True if released successfully
        """
        # Use Lua script to ensure atomic check-and-delete
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        
        result = self.redis.eval(lua_script, 1, self.lock_key, lock_token)
        return result == 1
    
    def extend_lock(self, lock_token: str, ttl: int = None) -> bool:
        """
        Extend lock TTL
        
        Args:
            lock_token: Token from acquire_lock
            ttl: New TTL in seconds (default: self.lock_ttl)
            
        Returns:
            True if extended successfully
        """
        if ttl is None:
            ttl = self.lock_ttl
        
        # Check if we still own the lock
        current_token = self.redis.get(self.lock_key)
        if current_token and current_token.decode() == lock_token:
            self.redis.expire(self.lock_key, ttl)
            return True
        
        return False
    
    def is_locked(self) -> bool:
        """Check if ingestion is currently locked"""
        return self.redis.exists(self.lock_key) > 0


class CacheInvalidationService:
    """
    Service for invalidating Redis caches after ingestion
    """
    
    def __init__(self, redis_client):
        """
        Initialize cache invalidation service
        
        Args:
            redis_client: Redis client
        """
        self.redis = redis_client
    
    def invalidate_all(self):
        """Invalidate all relevant caches after ingestion"""
        patterns = [
            'search:*',
            'ranking:*',
            'resource:*',
            'trending:*',
            'category:*',
        ]
        
        for pattern in patterns:
            self._invalidate_pattern(pattern)
    
    def _invalidate_pattern(self, pattern: str):
        """Delete all keys matching pattern"""
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            if keys:
                self.redis.delete(*keys)
            
            if cursor == 0:
                break
    
    def invalidate_search_cache(self):
        """Invalidate search-related caches"""
        self._invalidate_pattern('search:*')
    
    def invalidate_ranking_cache(self):
        """Invalidate ranking-related caches"""
        self._invalidate_pattern('ranking:*')
        self._invalidate_pattern('trending:*')
    
    def invalidate_resource_cache(self):
        """Invalidate resource-specific caches"""
        self._invalidate_pattern('resource:*')
