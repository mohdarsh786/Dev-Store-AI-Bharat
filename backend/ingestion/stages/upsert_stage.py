"""Upsert stage - inserts or updates resources in database."""

from __future__ import annotations

import logging
from typing import Dict, List

from models import CanonicalResource
from ingestion.repository import IngestionRepository

logger = logging.getLogger(__name__)


class UpsertStage:
    """Upserts resources into PostgreSQL database."""

    def __init__(self, repository: IngestionRepository):
        self.repository = repository

    def execute(self, resources: List[CanonicalResource]) -> Dict[str, any]:
        """
        Upsert resources into database.

        Args:
            resources: List of deduplicated canonical resources

        Returns:
            Dict with upserted resources and statistics
        """
        results = {
            "resources": [],
            "inserted": 0,
            "updated": 0,
            "unchanged": 0,
            "failed": 0,
        }

        for resource in resources:
            try:
                result = self.repository.upsert_resource(resource)

                # Track statistics
                if result.action == "inserted":
                    results["inserted"] += 1
                elif result.action == "updated":
                    results["updated"] += 1
                elif result.action == "unchanged":
                    results["unchanged"] += 1

                # Store resource with ID and embedding change flag
                resource_data = {
                    "id": result.resource_id,
                    "resource": resource,
                    "embedding_changed": result.embedding_changed,
                    "action": result.action,
                }
                results["resources"].append(resource_data)

            except Exception as e:
                logger.error(f"Failed to upsert resource {resource.name}: {e}", exc_info=True)
                results["failed"] += 1

        logger.info(
            f"Upsert stage complete: {results['inserted']} inserted, "
            f"{results['updated']} updated, {results['unchanged']} unchanged, "
            f"{results['failed']} failed"
        )

        return results
