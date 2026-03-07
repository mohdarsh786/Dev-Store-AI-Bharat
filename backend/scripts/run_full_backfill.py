"""
Full backfill script - runs complete ingestion across all sources.

This script:
1. Runs ingestion across all sources
2. Verifies Aurora records created
3. Verifies embeddings generated
4. Verifies OpenSearch documents indexed
5. Verifies ranking computed
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.database import DatabaseClient
from clients.opensearch import OpenSearchClient
from clients.redis_client import RedisClient
from ingestion.pipeline import run_ingestion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backfill.log"),
    ],
)

logger = logging.getLogger(__name__)


async def verify_infrastructure():
    """Verify all infrastructure is available."""
    logger.info("Verifying infrastructure...")

    try:
        # Check database
        db = DatabaseClient()
        db_health = db.health_check()
        if db_health["status"] != "healthy":
            raise Exception(f"Database unhealthy: {db_health}")
        logger.info("✓ Database healthy")

        # Check Redis
        redis = RedisClient()
        await redis.connect()
        if not await redis.ping():
            raise Exception("Redis ping failed")
        logger.info("✓ Redis healthy")
        await redis.disconnect()

        # Check OpenSearch
        os = OpenSearchClient()
        os_health = os.health_check()
        if os_health["status"] != "healthy":
            raise Exception(f"OpenSearch unhealthy: {os_health}")
        logger.info("✓ OpenSearch healthy")

        return True

    except Exception as e:
        logger.error(f"Infrastructure check failed: {e}")
        return False


async def verify_results(result: dict):
    """Verify ingestion results."""
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION")
    logger.info("=" * 70)

    stats = result.get("stats", {})

    # Check Aurora records
    logger.info(f"\n1. Aurora Records:")
    logger.info(f"   Inserted: {stats.get('inserted', 0)}")
    logger.info(f"   Updated: {stats.get('updated', 0)}")
    logger.info(f"   Total: {stats.get('inserted', 0) + stats.get('updated', 0)}")

    if stats.get("inserted", 0) + stats.get("updated", 0) == 0:
        logger.warning("   ⚠ No records created/updated in Aurora")
    else:
        logger.info("   ✓ Aurora records created")

    # Check embeddings
    logger.info(f"\n2. Embeddings:")
    logger.info(f"   Generated: {stats.get('embedded', 0)}")

    if stats.get("embedded", 0) == 0:
        logger.warning("   ⚠ No embeddings generated")
    else:
        logger.info("   ✓ Embeddings generated")

    # Check OpenSearch indexing
    logger.info(f"\n3. OpenSearch Documents:")
    logger.info(f"   Indexed: {stats.get('indexed', 0)}")

    if stats.get("indexed", 0) == 0:
        logger.warning("   ⚠ No documents indexed in OpenSearch")
    else:
        logger.info("   ✓ OpenSearch documents indexed")

    # Check rankings
    logger.info(f"\n4. Rankings:")
    logger.info(f"   Ranked: {stats.get('ranked', 0)}")

    if stats.get("ranked", 0) == 0:
        logger.warning("   ⚠ No rankings computed")
    else:
        logger.info("   ✓ Rankings computed")

    # Overall status
    logger.info(f"\n5. Overall Status:")
    logger.info(f"   Status: {result.get('status', 'unknown')}")
    logger.info(f"   Duration: {result.get('duration_seconds', 0):.2f} seconds")
    logger.info(f"   Failed: {stats.get('failed', 0)}")

    if result.get("status") == "success" and stats.get("failed", 0) == 0:
        logger.info("   ✓ Backfill completed successfully")
        return True
    else:
        logger.warning("   ⚠ Backfill completed with issues")
        return False


async def main():
    """Main entry point."""
    logger.info("=" * 70)
    logger.info("FULL BACKFILL STARTED")
    logger.info("=" * 70)

    # Verify infrastructure
    if not await verify_infrastructure():
        logger.error("Infrastructure verification failed. Aborting.")
        sys.exit(1)

    # Run ingestion
    logger.info("\nRunning ingestion across all sources...")
    result = await run_ingestion(sources=["github", "huggingface", "kaggle", "openrouter"])

    # Verify results
    success = await verify_results(result)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
