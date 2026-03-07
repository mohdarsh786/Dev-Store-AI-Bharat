"""Fetch stage - retrieves data from external sources."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ingestion.fetchers.github_fetcher import GitHubFetcher
from ingestion.fetchers.huggingface_fetcher import HuggingFaceFetcher
from ingestion.fetchers.kaggle_fetcher import KaggleFetcher
from ingestion.fetchers.openrouter_fetcher import OpenRouterFetcher

logger = logging.getLogger(__name__)


class FetchStage:
    """Fetches resources from external sources."""

    def __init__(self):
        self.fetchers = {
            "github": GitHubFetcher(),
            "huggingface": HuggingFaceFetcher(),
            "kaggle": KaggleFetcher(),
            "openrouter": OpenRouterFetcher(),
        }

    def execute(self, sources: List[str]) -> Dict[str, Any]:
        """
        Fetch data from specified sources.

        Args:
            sources: List of source names to fetch from

        Returns:
            Dict with fetched data and statistics
        """
        results = {
            "models": [],
            "datasets": [],
            "repositories": [],
            "stats": {},
        }

        for source in sources:
            if source not in self.fetchers:
                logger.warning(f"Unknown source: {source}")
                continue

            try:
                logger.info(f"Fetching from {source}...")
                fetcher = self.fetchers[source]

                if source == "github":
                    repos = fetcher.fetch_and_normalize_all()
                    results["repositories"].extend(repos)
                    results["stats"][source] = {"repositories": len(repos)}
                    logger.info(f"✓ {source}: {len(repos)} repositories")

                elif source == "huggingface":
                    data = fetcher.fetch_and_normalize_all()
                    results["models"].extend(data.get("models", []))
                    results["datasets"].extend(data.get("datasets", []))
                    results["stats"][source] = {
                        "models": len(data.get("models", [])),
                        "datasets": len(data.get("datasets", [])),
                    }
                    logger.info(
                        f"✓ {source}: {len(data.get('models', []))} models, "
                        f"{len(data.get('datasets', []))} datasets"
                    )

                elif source == "kaggle":
                    datasets = fetcher.fetch_and_normalize_all(max_pages=5)
                    results["datasets"].extend(datasets)
                    results["stats"][source] = {"datasets": len(datasets)}
                    logger.info(f"✓ {source}: {len(datasets)} datasets")

                elif source == "openrouter":
                    models = fetcher.fetch_and_normalize_all()
                    results["models"].extend(models)
                    results["stats"][source] = {"models": len(models)}
                    logger.info(f"✓ {source}: {len(models)} models")

            except Exception as e:
                logger.error(f"✗ Failed to fetch from {source}: {e}", exc_info=True)
                results["stats"][source] = {"error": str(e)}

        total = len(results["models"]) + len(results["datasets"]) + len(results["repositories"])
        logger.info(f"Fetch stage complete: {total} resources fetched")

        return results
