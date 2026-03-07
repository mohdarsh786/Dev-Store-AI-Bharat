"""Normalize stage - converts fetched data to canonical schema."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from models import CanonicalResource, IngestionSource
from ingestion.normalization import canonicalize_resource

logger = logging.getLogger(__name__)


class NormalizeStage:
    """Normalizes fetched resources to canonical schema."""

    def execute(self, fetch_results: Dict[str, Any]) -> List[CanonicalResource]:
        """
        Normalize all fetched resources to canonical schema.

        Args:
            fetch_results: Output from FetchStage

        Returns:
            List of CanonicalResource objects
        """
        normalized = []

        # Process models
        for model in fetch_results.get("models", []):
            try:
                source = IngestionSource(model["source"])
                canonical = canonicalize_resource(source, model, model)
                normalized.append(canonical)
            except Exception as e:
                logger.error(f"Failed to normalize model {model.get('name')}: {e}")

        # Process datasets
        for dataset in fetch_results.get("datasets", []):
            try:
                source = IngestionSource(dataset["source"])
                canonical = canonicalize_resource(source, dataset, dataset)
                normalized.append(canonical)
            except Exception as e:
                logger.error(f"Failed to normalize dataset {dataset.get('name')}: {e}")

        # Process repositories
        for repo in fetch_results.get("repositories", []):
            try:
                source = IngestionSource(repo["source"])
                canonical = canonicalize_resource(source, repo, repo)
                normalized.append(canonical)
            except Exception as e:
                logger.error(f"Failed to normalize repository {repo.get('name')}: {e}")

        logger.info(f"Normalize stage complete: {len(normalized)} resources normalized")
        return normalized
