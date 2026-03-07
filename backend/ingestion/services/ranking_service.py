"""
Ranking Service

Computes ranking scores for resources
"""
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any


class RankingService:
    """
    Service for computing resource rankings
    
    Computes:
    - rank_score: Overall popularity score
    - trending_score: Recent activity score
    - category_rank: Rank within category
    """
    
    def __init__(self, db_client):
        """
        Initialize ranking service
        
        Args:
            db_client: Database client
        """
        self.db = db_client
    
    def compute_all_rankings(self) -> Dict[str, int]:
        """
        Compute rankings for all resources
        
        Returns:
            Statistics: updated count
        """
        stats = {'updated': 0}
        
        # Get all resources
        resources = self._get_all_resources()
        
        # Compute scores
        for resource in resources:
            rank_score = self._compute_rank_score(resource)
            trending_score = self._compute_trending_score(resource)
            
            # Update resource
            self._update_scores(
                resource['id'],
                rank_score,
                trending_score
            )
            stats['updated'] += 1
        
        # Compute category ranks
        self._compute_category_ranks()
        
        return stats
    
    def _get_all_resources(self) -> List[Dict[str, Any]]:
        """Get all resources for ranking"""
        query = """
            SELECT
                id,
                stars,
                downloads,
                created_at,
                updated_at,
                category
            FROM resources
            ORDER BY id
        """
        
        return self.db.fetch_all(query)
    
    def _compute_rank_score(self, resource: Dict[str, Any]) -> float:
        """
        Compute overall rank score
        
        Formula: log(stars + 1) + log(downloads + 1) * 0.5
        """
        stars = resource.get('stars', 0)
        downloads = resource.get('downloads', 0)
        
        score = math.log(stars + 1) + math.log(downloads + 1) * 0.5
        return round(score, 2)
    
    def _compute_trending_score(self, resource: Dict[str, Any]) -> float:
        """
        Compute trending score based on recency
        
        Formula: rank_score * recency_factor
        Recency factor decays over time
        """
        rank_score = self._compute_rank_score(resource)
        
        # Calculate days since last update
        updated_at = resource.get('updated_at')
        if not updated_at:
            return rank_score * 0.1
        
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        days_old = (datetime.utcnow() - updated_at).days
        
        # Recency factor: 1.0 for new, decays to 0.1 over 365 days
        recency_factor = max(0.1, 1.0 - (days_old / 365.0))
        
        trending_score = rank_score * recency_factor
        return round(trending_score, 2)
    
    def _update_scores(
        self,
        resource_id: str,
        rank_score: float,
        trending_score: float
    ):
        """Update resource with computed scores"""
        query = """
            UPDATE resources
            SET
                rank_score = %s,
                trending_score = %s,
                ranking_updated_at = NOW()
            WHERE id = %s
        """
        
        self.db.execute(query, (rank_score, trending_score, resource_id))
    
    def _compute_category_ranks(self):
        """
        Compute rank within each category
        
        Uses ROW_NUMBER() to assign ranks
        """
        query = """
            WITH ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        PARTITION BY category
                        ORDER BY rank_score DESC
                    ) as category_rank
                FROM resources
            )
            UPDATE resources r
            SET category_rank = ranked.category_rank
            FROM ranked
            WHERE r.id = ranked.id
        """
        
        self.db.execute(query)
