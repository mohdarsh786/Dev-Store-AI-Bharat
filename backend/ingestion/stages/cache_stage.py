"""Cache stage - invalidates Redis caches after ingestion."""

from __future__ import annotations

import logging
from typing import Dict

from clients.redis_client import RedisClient

logger = logging.getLogger(__name__)


class CacheStage:
    """Invalidates Redis caches after successful ingestion."""

    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client

    async def execute(self) -> Dict[str, int]:
        """
        Invalidate all relevant caches.

        Returns:
            Dict with invalidation statistics
        """
        logger.info("Invalidating caches...")

        results = {
            "search": 0,
            "ranking": 0,
            "resource": 0,
            "trending": 0,
        }

        try:
            # Invalidate search caches
            results["search"] = await self.redis.invalidate_pattern("search:*")

            # Invalidate ranking caches
            results["ranking"] = await self.redis.invalidate_pattern("ranking:*")

            # Invalidate resource caches
            results["resource"] = await self.redis.invalidate_pattern("resource:*")

            # Invalidate trending caches
            results["trending"] = await self.redis.invalidate_pattern("trending:*")

            total = sum(results.values())
            logger.info(f"Cache stage complete: {total} keys invalidated")

        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}", exc_info=True)

        return results
