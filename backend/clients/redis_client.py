"""
Redis Client for caching and fast data access
Supports AWS ElastiCache Redis with connection pooling and retry logic
"""
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from typing import Optional, Any, List
import json
import logging
from datetime import timedelta
import uuid

from config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Redis client with connection pooling and error handling
    
    Key Naming Strategy:
    - search:{query_hash} - Search results cache
    - ranking:{resource_id}:{date} - Daily ranking scores
    - resource:{resource_id} - Resource metadata
    - user:{user_id}:profile - User profile data
    - health:{resource_id} - Health check status
    - embedding:{text_hash} - Cached embeddings
    
    TTL Strategy:
    - Search results: 5 minutes (300s)
    - Rankings: 1 hour (3600s)
    - Resource metadata: 15 minutes (900s)
    - User profiles: 30 minutes (1800s)
    - Health status: 10 minutes (600s)
    - Embeddings: 24 hours (86400s)
    """
    
    def __init__(self):
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self):
        """Establish connection to Redis with retry logic"""
        if self._connected:
            return
        
        try:
            # Create connection pool
            self.pool = ConnectionPool(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password if settings.redis_password else None,
                db=settings.redis_db,
                decode_responses=True,
                max_connections=settings.redis_pool_size,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30
            )
            
            # Create Redis client
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            
            self._connected = True
            logger.info(f"Connected to Redis at {settings.redis_host}:{settings.redis_port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            await self.pool.disconnect()
            self._connected = False
            logger.info("Disconnected from Redis")
    
    async def ping(self) -> bool:
        """Health check"""
        try:
            return await self.client.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {str(e)}")
            return False
    
    # ==================== Generic Operations ====================
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        try:
            return await self.client.get(key)
        except Exception as e:
            logger.error(f"Redis GET failed for key {key}: {str(e)}")
            return None
    
    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value with optional TTL"""
        try:
            if ttl:
                return await self.client.setex(key, ttl, value)
            else:
                return await self.client.set(key, value)
        except Exception as e:
            logger.error(f"Redis SET failed for key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key"""
        try:
            return await self.client.delete(key) > 0
        except Exception as e:
            logger.error(f"Redis DELETE failed for key {key}: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS failed for key {key}: {str(e)}")
            return False

    async def incrby(self, key: str, amount: int = 1) -> int:
        """Increment a numeric key by the requested amount."""
        try:
            return await self.client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis INCRBY failed for key {key}: {str(e)}")
            return 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on existing key"""
        try:
            return await self.client.expire(key, ttl)
        except Exception as e:
            logger.error(f"Redis EXPIRE failed for key {key}: {str(e)}")
            return False
    
    # ==================== JSON Operations ====================
    
    async def get_json(self, key: str) -> Optional[dict]:
        """Get JSON value"""
        try:
            value = await self.get(key)
            return json.loads(value) if value else None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON for key {key}: {str(e)}")
            return None
    
    async def set_json(
        self,
        key: str,
        value: dict,
        ttl: Optional[int] = None
    ) -> bool:
        """Set JSON value"""
        try:
            json_str = json.dumps(value)
            return await self.set(key, json_str, ttl)
        except Exception as e:
            logger.error(f"Failed to set JSON for key {key}: {str(e)}")
            return False
    
    # ==================== Search Cache ====================
    
    async def cache_search_results(
        self,
        query_hash: str,
        results: dict,
        ttl: int = 300
    ) -> bool:
        """Cache search results (5 min TTL)"""
        key = f"search:{query_hash}"
        return await self.set_json(key, results, ttl)
    
    async def get_cached_search(self, query_hash: str) -> Optional[dict]:
        """Get cached search results"""
        key = f"search:{query_hash}"
        return await self.get_json(key)
    
    # ==================== Ranking Cache ====================
    
    async def cache_ranking_score(
        self,
        resource_id: str,
        date: str,
        score: dict,
        ttl: int = 3600
    ) -> bool:
        """Cache ranking score (1 hour TTL)"""
        key = f"ranking:{resource_id}:{date}"
        return await self.set_json(key, score, ttl)
    
    async def get_cached_ranking(
        self,
        resource_id: str,
        date: str
    ) -> Optional[dict]:
        """Get cached ranking score"""
        key = f"ranking:{resource_id}:{date}"
        return await self.get_json(key)
    
    async def invalidate_ranking(self, resource_id: str) -> int:
        """Invalidate all ranking caches for a resource"""
        pattern = f"ranking:{resource_id}:*"
        keys = await self.client.keys(pattern)
        if keys:
            return await self.client.delete(*keys)
        return 0
    
    # ==================== Resource Cache ====================
    
    async def cache_resource(
        self,
        resource_id: str,
        resource: dict,
        ttl: int = 900
    ) -> bool:
        """Cache resource metadata (15 min TTL)"""
        key = f"resource:{resource_id}"
        return await self.set_json(key, resource, ttl)
    
    async def get_cached_resource(self, resource_id: str) -> Optional[dict]:
        """Get cached resource"""
        key = f"resource:{resource_id}"
        return await self.get_json(key)
    
    async def invalidate_resource(self, resource_id: str) -> bool:
        """Invalidate resource cache"""
        key = f"resource:{resource_id}"
        return await self.delete(key)
    
    # ==================== User Profile Cache ====================
    
    async def cache_user_profile(
        self,
        user_id: str,
        profile: dict,
        ttl: int = 1800
    ) -> bool:
        """Cache user profile (30 min TTL)"""
        key = f"user:{user_id}:profile"
        return await self.set_json(key, profile, ttl)
    
    async def get_cached_user_profile(self, user_id: str) -> Optional[dict]:
        """Get cached user profile"""
        key = f"user:{user_id}:profile"
        return await self.get_json(key)
    
    async def invalidate_user_profile(self, user_id: str) -> bool:
        """Invalidate user profile cache"""
        key = f"user:{user_id}:profile"
        return await self.delete(key)
    
    # ==================== Health Status Cache ====================
    
    async def cache_health_status(
        self,
        resource_id: str,
        status: dict,
        ttl: int = 600
    ) -> bool:
        """Cache health status (10 min TTL)"""
        key = f"health:{resource_id}"
        return await self.set_json(key, status, ttl)
    
    async def get_cached_health_status(self, resource_id: str) -> Optional[dict]:
        """Get cached health status"""
        key = f"health:{resource_id}"
        return await self.get_json(key)
    
    # ==================== Embedding Cache ====================
    
    async def cache_embedding(
        self,
        text_hash: str,
        embedding: List[float],
        ttl: int = 86400
    ) -> bool:
        """Cache embedding vector (24 hour TTL)"""
        key = f"embedding:{text_hash}"
        return await self.set_json(key, {"embedding": embedding}, ttl)
    
    async def get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embedding"""
        key = f"embedding:{text_hash}"
        data = await self.get_json(key)
        return data.get("embedding") if data else None
    
    # ==================== Cache Invalidation ====================
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        try:
            keys = await self.client.keys(pattern)
            if keys:
                return await self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Failed to invalidate pattern {pattern}: {str(e)}")
            return 0

    async def acquire_lock(self, key: str, ttl: int) -> Optional[str]:
        """Acquire a distributed lock using SET NX EX semantics."""
        token = str(uuid.uuid4())
        try:
            acquired = await self.client.set(key, token, ex=ttl, nx=True)
            return token if acquired else None
        except Exception as e:
            logger.error(f"Failed to acquire lock {key}: {str(e)}")
            return None

    async def release_lock(self, key: str, token: str) -> bool:
        """Release a lock only if the stored token matches."""
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        else
            return 0
        end
        """
        try:
            result = await self.client.eval(script, 1, key, token)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to release lock {key}: {str(e)}")
            return False
    
    async def flush_all(self) -> bool:
        """Flush all cache (use with caution!)"""
        try:
            await self.client.flushdb()
            logger.warning("Redis cache flushed")
            return True
        except Exception as e:
            logger.error(f"Failed to flush cache: {str(e)}")
            return False
    
    # ==================== Statistics ====================
    
    async def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        try:
            info = await self.client.info("stats")
            return {
                "total_connections": info.get("total_connections_received", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {}
    
    @staticmethod
    def _calculate_hit_rate(hits: int, misses: int) -> float:
        """Calculate cache hit rate"""
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0
