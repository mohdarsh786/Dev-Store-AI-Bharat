"""Ranking stage - computes and updates resource rankings."""

from __future__ import annotations

import logging
from typing import Dict

from ingestion.repository import IngestionRepository

logger = logging.getLogger(__name__)


class RankingStage:
    """Computes and updates resource rankings."""

    def __init__(self, repository: IngestionRepository):
        self.repository = repository

    def execute(self) -> Dict[str, any]:
        """
        Compute and update rankings for all resources.

        Returns:
            Dict with ranking statistics
        """
        logger.info("Computing rankings...")

        try:
            # Fetch all resources for ranking
            resources = self.repository.fetch_resources_for_ranking()

            if not resources:
                logger.warning("No resources found for ranking")
                return {"ranked": 0, "categories": 0}

            # Compute rankings
            rankings = []
            categories = set()

            for resource in resources:
                rank_score = self._compute_rank_score(resource)
                trending_score = self._compute_trending_score(resource)
                category = resource.get("type", "unknown")
                categories.add(category)

                rankings.append(
                    {
                        "id": resource["id"],
                        "rank_score": rank_score,
                        "trending_score": trending_score,
                        "category_rank": 0,  # Will be computed in persist_rankings
                        "popularity_score": rank_score,
                        "optimization_score": 0.5,
                        "freshness_score": trending_score / rank_score if rank_score > 0 else 0,
                        "final_score": rank_score,
                        "rank_position": 0,
                    }
                )

            # Sort by rank_score
            rankings.sort(key=lambda x: x["rank_score"], reverse=True)

            # Assign positions
            for i, ranking in enumerate(rankings):
                ranking["rank_position"] = i + 1

            # Persist rankings
            result = self.repository.persist_rankings(rankings)

            logger.info(
                f"Ranking stage complete: {result['ranked_count']} resources ranked, "
                f"{len(categories)} categories"
            )

            return {
                "ranked": result["ranked_count"],
                "categories": len(categories),
                "top_ranked_id": result.get("top_ranked_id"),
            }

        except Exception as e:
            logger.error(f"Ranking stage failed: {e}", exc_info=True)
            return {"ranked": 0, "categories": 0, "error": str(e)}

    def _compute_rank_score(self, resource: Dict) -> float:
        """Compute overall rank score based on popularity."""
        import math

        stars = resource.get("github_stars", 0) or 0
        downloads = resource.get("download_count", 0) or 0

        score = math.log(stars + 1) + math.log(downloads + 1) * 0.5
        return round(score, 2)

    def _compute_trending_score(self, resource: Dict) -> float:
        """Compute trending score based on recency."""
        from datetime import datetime

        rank_score = self._compute_rank_score(resource)

        updated_at = resource.get("updated_at")
        if not updated_at:
            return rank_score * 0.1

        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except ValueError:
                return rank_score * 0.1

        days_old = (datetime.utcnow() - updated_at).days
        recency_factor = max(0.1, 1.0 - (days_old / 365.0))

        return round(rank_score * recency_factor, 2)
