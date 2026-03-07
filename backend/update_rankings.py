"""Refresh ranking scores from persisted resources."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from clients.redis_client import RedisClient
from config import settings
from ingestion.repository import IngestionRepository
from services.ranking import RankingService


async def _ingestion_is_running() -> bool:
    redis_client = RedisClient()
    await redis_client.connect()
    try:
        return await redis_client.exists(settings.ingestion_lock_key)
    finally:
        await redis_client.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh rank_score, trending_score, and category_rank")
    parser.add_argument(
        "--skip-if-ingestion-active",
        action="store_true",
        help="Exit without updating rankings when the ingestion lock is held",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.skip_if_ingestion_active and asyncio.run(_ingestion_is_running()):
        print(json.dumps({"status": "skipped", "reason": "ingestion_lock_active"}, indent=2))
        return 0

    repository = IngestionRepository()
    ranking_service = RankingService()

    refreshed = ranking_service.refresh_rankings(repository.fetch_resources_for_ranking())
    stats = repository.persist_rankings(refreshed)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
