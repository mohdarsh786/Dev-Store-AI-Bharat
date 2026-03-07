"""Production ingestion pipeline orchestrator."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from clients.bedrock import BedrockClient
from clients.database import DatabaseClient
from clients.opensearch import OpenSearchClient
from clients.redis_client import RedisClient
from ingestion.repository import IngestionRepository
from ingestion.services.lock_service import LockService
from ingestion.services.run_tracker import RunTracker
from ingestion.snapshots import SnapshotStore
from ingestion.stages import (
    CacheStage,
    DedupeStage,
    EmbeddingStage,
    FetchStage,
    IndexingStage,
    NormalizeStage,
    RankingStage,
    UpsertStage,
)
from models import IngestionStatus

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Production ingestion pipeline orchestrator.

    Pipeline flow:
    1. Acquire lock
    2. Fetch from sources
    3. Normalize to canonical schema
    4. Deduplicate
    5. Upsert to database
    6. Generate embeddings
    7. Index in OpenSearch
    8. Refresh rankings
    9. Invalidate caches
    10. Release lock
    """

    def __init__(
        self,
        db_client: DatabaseClient,
        bedrock_client: BedrockClient,
        opensearch_client: OpenSearchClient,
        redis_client: RedisClient,
        run_id: Optional[str] = None,
    ):
        self.run_id = run_id or str(uuid.uuid4())
        self.started_at = datetime.utcnow()
        self.finished_at = None

        # Clients
        self.db = db_client
        self.bedrock = bedrock_client
        self.opensearch = opensearch_client
        self.redis = redis_client

        # Services
        self.repository = IngestionRepository(db_client)
        self.lock_service = LockService(redis_client)
        self.run_tracker = RunTracker(db_client)
        self.snapshot_store = SnapshotStore()

        # Stages
        self.fetch_stage = FetchStage()
        self.normalize_stage = NormalizeStage()
        self.dedupe_stage = DedupeStage()
        self.upsert_stage = UpsertStage(self.repository)
        self.embedding_stage = EmbeddingStage(bedrock_client, redis_client, self.repository)
        self.indexing_stage = IndexingStage(opensearch_client, self.repository)
        self.ranking_stage = RankingStage(self.repository)
        self.cache_stage = CacheStage(redis_client)

        # State
        self.lock_token = None
        self.stats = {
            "fetched": 0,
            "normalized": 0,
            "deduplicated": 0,
            "inserted": 0,
            "updated": 0,
            "unchanged": 0,
            "embedded": 0,
            "indexed": 0,
            "ranked": 0,
            "failed": 0,
        }

        logger.info(f"Pipeline initialized with run_id: {self.run_id}")

    async def run(self, sources: List[str]) -> Dict[str, Any]:
        """
        Execute the complete ingestion pipeline.

        Args:
            sources: List of source names to ingest from

        Returns:
            Dict with run results and statistics
        """
        try:
            logger.info("=" * 70)
            logger.info(f"INGESTION PIPELINE STARTED - Run ID: {self.run_id}")
            logger.info(f"Sources: {', '.join(sources)}")
            logger.info("=" * 70)

            # Create run record
            for source in sources:
                self.run_tracker.create_run(self.run_id, source, IngestionStatus.RUNNING)

            # Stage 0: Acquire lock
            if not await self._acquire_lock():
                return self._build_result(IngestionStatus.SKIPPED, "Lock already held")

            # Stage 1: Fetch
            fetch_results = self.fetch_stage.execute(sources)
            self.stats["fetched"] = (
                len(fetch_results["models"])
                + len(fetch_results["datasets"])
                + len(fetch_results["repositories"])
            )

            # Save raw snapshots
            await self._save_snapshots(sources, fetch_results)

            # Stage 2: Normalize
            normalized = self.normalize_stage.execute(fetch_results)
            self.stats["normalized"] = len(normalized)

            # Stage 3: Deduplicate
            deduplicated = self.dedupe_stage.execute(normalized)
            self.stats["deduplicated"] = len(deduplicated)

            # Stage 4: Upsert
            upsert_results = self.upsert_stage.execute(deduplicated)
            self.stats["inserted"] = upsert_results["inserted"]
            self.stats["updated"] = upsert_results["updated"]
            self.stats["unchanged"] = upsert_results["unchanged"]
            self.stats["failed"] += upsert_results["failed"]

            # Stage 5: Embeddings
            embedding_results = await self.embedding_stage.execute(upsert_results)
            self.stats["embedded"] = embedding_results["stats"]["embedded"]
            self.stats["failed"] += embedding_results["stats"]["failed"]

            # Stage 6: Indexing
            indexing_results = self.indexing_stage.execute(embedding_results)
            self.stats["indexed"] = indexing_results["indexed"]
            self.stats["failed"] += indexing_results["failed"]

            # Stage 7: Rankings
            ranking_results = self.ranking_stage.execute()
            self.stats["ranked"] = ranking_results["ranked"]

            # Stage 8: Cache invalidation
            await self.cache_stage.execute()

            # Update run records
            for source in sources:
                self.run_tracker.update_run(
                    self.run_id,
                    source,
                    status=IngestionStatus.SUCCESS,
                    counters={
                        "fetched_count": self.stats["fetched"],
                        "inserted_count": self.stats["inserted"],
                        "updated_count": self.stats["updated"],
                        "unchanged_count": self.stats["unchanged"],
                        "failed_count": self.stats["failed"],
                        "embedded_count": self.stats["embedded"],
                        "indexed_count": self.stats["indexed"],
                    },
                    finished=True,
                )

            return self._build_result(IngestionStatus.SUCCESS)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)

            # Update run records
            for source in sources:
                self.run_tracker.update_run(
                    self.run_id,
                    source,
                    status=IngestionStatus.FAILED,
                    top_failure_reason=str(e),
                    finished=True,
                )

            return self._build_result(IngestionStatus.FAILED, str(e))

        finally:
            # Release lock
            if self.lock_token:
                await self._release_lock()

            self.finished_at = datetime.utcnow()
            duration = (self.finished_at - self.started_at).total_seconds()

            logger.info("=" * 70)
            logger.info("INGESTION PIPELINE COMPLETED")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Fetched: {self.stats['fetched']}")
            logger.info(f"Inserted: {self.stats['inserted']}")
            logger.info(f"Updated: {self.stats['updated']}")
            logger.info(f"Embedded: {self.stats['embedded']}")
            logger.info(f"Indexed: {self.stats['indexed']}")
            logger.info(f"Ranked: {self.stats['ranked']}")
            logger.info(f"Failed: {self.stats['failed']}")
            logger.info("=" * 70)

    async def _acquire_lock(self) -> bool:
        """Acquire distributed lock."""
        logger.info("Acquiring ingestion lock...")

        try:
            await self.redis.connect()
            self.lock_token = self.lock_service.acquire_lock(timeout=10)

            if self.lock_token:
                logger.info("✓ Lock acquired")
                return True
            else:
                logger.warning("✗ Lock already held by another process")
                return False

        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return False

    async def _release_lock(self):
        """Release distributed lock."""
        logger.info("Releasing ingestion lock...")

        try:
            if self.lock_service.release_lock(self.lock_token):
                logger.info("✓ Lock released")
            else:
                logger.warning("✗ Failed to release lock (may have expired)")

        except Exception as e:
            logger.error(f"Error releasing lock: {e}")

    async def _save_snapshots(self, sources: List[str], fetch_results: Dict[str, Any]):
        """Save raw snapshots to S3."""
        try:
            for source in sources:
                if source in fetch_results.get("stats", {}):
                    # Combine all data for this source
                    source_data = {
                        "models": [
                            m for m in fetch_results.get("models", []) if m.get("source") == source
                        ],
                        "datasets": [
                            d for d in fetch_results.get("datasets", []) if d.get("source") == source
                        ],
                        "repositories": [
                            r
                            for r in fetch_results.get("repositories", [])
                            if r.get("source") == source
                        ],
                    }

                    self.snapshot_store.persist(source, self.run_id, source_data)

        except Exception as e:
            logger.warning(f"Failed to save snapshots: {e}")

    def _build_result(self, status: IngestionStatus, error: Optional[str] = None) -> Dict[str, Any]:
        """Build result dictionary."""
        result = {
            "run_id": self.run_id,
            "status": status.value,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": (
                (self.finished_at - self.started_at).total_seconds() if self.finished_at else 0
            ),
            "stats": self.stats,
        }

        if error:
            result["error"] = error

        return result


async def run_ingestion(sources: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to run ingestion pipeline.

    Args:
        sources: List of sources to ingest from (default: all)

    Returns:
        Dict with run results
    """
    if sources is None:
        sources = ["github", "huggingface", "kaggle", "openrouter"]

    # Initialize clients
    db_client = DatabaseClient()
    bedrock_client = BedrockClient()
    opensearch_client = OpenSearchClient()
    redis_client = RedisClient()

    # Create and run pipeline
    pipeline = IngestionPipeline(
        db_client=db_client,
        bedrock_client=bedrock_client,
        opensearch_client=opensearch_client,
        redis_client=redis_client,
    )

    return await pipeline.run(sources)
