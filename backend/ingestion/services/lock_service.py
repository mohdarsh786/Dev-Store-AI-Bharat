"""
Lock Service

In-memory locking for ingestion coordination
"""
import asyncio


class LockService:
    """
    In-memory lock service using asyncio
    
    Prevents overlapping ingestion runs on a single node
    """
    
    def __init__(self):
        """Initialize lock service"""
        self._lock = asyncio.Lock()
    
    async def acquire_lock(self) -> bool:
        """
        Acquire ingestion lock
        
        Returns:
            True if acquired, False otherwise
        """
        if self._lock.locked():
            return False
            
        await self._lock.acquire()
        return True
    
    def release_lock(self):
        """
        Release ingestion lock
        """
        if self._lock.locked():
            self._lock.release()
            return True
            
        return False
    
    def is_locked(self) -> bool:
        """Check if ingestion is currently locked"""
        return self._lock.locked()


class CacheInvalidationService:
    """
    Service for invalidating caches after ingestion
    """
    
    def __init__(self, target_cache):
        """
        Initialize cache invalidation service
        
        Args:
            target_cache: Target cachetools TTLCache reference
        """
        self._cache = target_cache
    
    def invalidate_all(self):
        """Invalidate all relevant caches after ingestion"""
        if self._cache:
            self._cache.clear()
    
    def invalidate_search_cache(self):
        """Invalidate search-related caches"""
        self._invalidate_prefix('search:')
            
    def _invalidate_prefix(self, prefix: str):
        if not self._cache:
            return
            
        keys_to_delete = [
            k for k in self._cache.keys()
            if isinstance(k, str) and k.startswith(prefix)
        ]
        
        for k in keys_to_delete:
            del self._cache[k]
