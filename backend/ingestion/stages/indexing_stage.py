"""Indexing stage - indexes resources in OpenSearch."""

from __future__ import annotations

import logging
from typing import Any, Dict

from clients.opensearch import OpenSearchClient

logger = logging.getLogger(__name__)


class IndexingStage:
    """Indexes resources in OpenSearch."""

    def __init__(self, opensearch_client: OpenSearchClient, repository):
        self.opensearch = opensearch_client
        self.repository = repository
        self.index_name = "devstore_resources"

    def execute(self, embedding_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Index resources in OpenSearch.

        Args:
            embedding_results: Output from EmbeddingStage

        Returns:
            Dict with indexing statistics
        """
        results = {
            "indexed": 0,
            "failed": 0,
            "skipped": 0,
        }

        resources = embedding_results.get("resources", [])
        resources_with_embeddings = [r for r in resources if r.get("embedding")]

        logger.info(f"Resources to index: {len(resources_with_embeddings)}")

        # Batch indexing
        batch_size = 100
        for i in range(0, len(resources_with_embeddings), batch_size):
            batch = resources_with_embeddings[i : i + batch_size]

            for resource_data in batch:
                try:
                    resource = resource_data["resource"]
                    resource_id = resource_data["id"]
                    embedding = resource_data.get("embedding", [])

                    # Create OpenSearch document
                    document = {
                        "id": resource_id,
                        "name": resource.name,
                        "description": resource.description,
                        "source": resource.source.value,
                        "source_url": resource.source_url,
                        "resource_type": resource.resource_type.value,
                        "pricing_type": resource.pricing_type.value,
                        "github_stars": resource.github_stars,
                        "download_count": resource.download_count,
                        "tags": resource.tags,
                        "categories": resource.categories,
                        "health_status": resource.health_status.value,
                        "embedding": embedding,
                    }

                    # Index document
                    self.opensearch.index_document(
                        document=document, doc_id=resource_id, index_name=self.index_name
                    )

                    # Mark as indexed in database
                    self.repository.mark_indexed(resource_id)

                    results["indexed"] += 1

                except Exception as e:
                    logger.error(
                        f"Failed to index resource {resource_data.get('id')}: {e}", exc_info=True
                    )
                    results["failed"] += 1

        logger.info(f"Indexing stage complete: {results['indexed']} indexed, {results['failed']} failed")

        return results
