"""Embedding stage - generates embeddings for resources."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List

from clients.bedrock import BedrockClient
from clients.redis_client import RedisClient

logger = logging.getLogger(__name__)


class EmbeddingStage:
    """Generates embeddings for new or changed resources."""

    def __init__(self, bedrock_client: BedrockClient, redis_client: RedisClient, repository):
        self.bedrock = bedrock_client
        self.redis = redis_client
        self.repository = repository

    async def execute(self, upsert_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate embeddings for resources that need them.

        Args:
            upsert_results: Output from UpsertStage

        Returns:
            Dict with embedding statistics
        """
        results = {
            "embedded": 0,
            "cached": 0,
            "failed": 0,
            "skipped": 0,
        }

        resources_to_embed = [
            r for r in upsert_results["resources"] if r.get("embedding_changed", True)
        ]

        logger.info(f"Resources needing embeddings: {len(resources_to_embed)}")

        for resource_data in resources_to_embed:
            resource = resource_data["resource"]
            resource_id = resource_data["id"]

            try:
                # Generate embedding text
                embedding_text = resource.embedding_text()
                text_hash = hashlib.sha256(embedding_text.encode()).hexdigest()

                # Check cache
                cached_embedding = await self.redis.get_cached_embedding(text_hash)

                if cached_embedding:
                    embedding = cached_embedding
                    results["cached"] += 1
                    logger.debug(f"Using cached embedding for {resource.name}")
                else:
                    # Generate via Bedrock
                    embedding = self.bedrock.generate_embedding(embedding_text)

                    # Cache it (30 days)
                    await self.redis.cache_embedding(text_hash, embedding, ttl=86400 * 30)
                    results["embedded"] += 1
                    logger.debug(f"Generated embedding for {resource.name}")

                # Update in database
                embedding_hash = resource.embedding_hash()
                self.repository.update_embedding(resource_id, embedding, embedding_hash)

                # Store embedding in resource data for indexing stage
                resource_data["embedding"] = embedding

            except Exception as e:
                logger.error(f"Failed to generate embedding for {resource.name}: {e}", exc_info=True)
                results["failed"] += 1

        logger.info(
            f"Embedding stage complete: {results['embedded']} generated, "
            f"{results['cached']} from cache, {results['failed']} failed"
        )

        return {
            "resources": upsert_results["resources"],
            "stats": results,
        }
