"""Dedupe stage - removes duplicate resources."""

from __future__ import annotations

import logging
from typing import Dict, List

from models import CanonicalResource

logger = logging.getLogger(__name__)


class DedupeStage:
    """Deduplicates resources based on unique_key (source + source_url)."""

    def execute(self, resources: List[CanonicalResource]) -> List[CanonicalResource]:
        """
        Remove duplicate resources.

        Args:
            resources: List of normalized resources

        Returns:
            Deduplicated list of resources
        """
        seen = {}
        deduplicated = []

        for resource in resources:
            unique_key = resource.unique_key

            if unique_key not in seen:
                seen[unique_key] = resource
                deduplicated.append(resource)
            else:
                logger.debug(f"Skipping duplicate: {resource.name} ({unique_key})")

        removed = len(resources) - len(deduplicated)
        logger.info(
            f"Dedupe stage complete: {len(resources)} → {len(deduplicated)} "
            f"({removed} duplicates removed)"
        )

        return deduplicated
