"""Cache stage - invalidates local caches after ingestion."""

from __future__ import annotations

import logging
from typing import Dict
from cachetools import TTLCache

logger = logging.getLogger(__name__)


class CacheStage:
    """Invalidates in-memory caches after successful ingestion."""

    def __init__(self, local_cache: TTLCache):
        self.cache = local_cache

    async def execute(self) -> Dict[str, int]:
        """
        Invalidate all relevant caches.

        Returns:
            Dict with invalidation statistics
        """
        logger.info("Invalidating caches...")
        keys_count = len(self.cache)

        try:
            self.cache.clear()
            logger.info(f"Cache stage complete: {keys_count} keys invalidated")

        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}", exc_info=True)

        return {"cleared": keys_count}
