"""Ranking service for DevStore"""
import math
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RankingService:
    """Service for computing resource ranking scores with validation"""
    
    def __init__(self):
        self.max_stars = 100000
        self.max_downloads = 1000000
        self.max_users = 10000
        self.max_latency_ms = 1000
        self.max_cost = 0.01
    
    def compute_semantic_relevance(self, cosine_similarity: float) -> float:
        """
        Compute semantic relevance score from cosine similarity.
        
        Args:
            cosine_similarity: Similarity score from vector search (typically 0-1)
            
        Returns:
            Normalized score in range [0, 1]
        """
        score = max(0.0, min(1.0, cosine_similarity))
        logger.debug(f"Semantic relevance: {cosine_similarity} -> {score}")
        return score
    
    def compute_popularity(
        self,
        github_stars: int = 0,
        downloads: int = 0,
        users: int = 0
    ) -> float:
        """
        Compute popularity score with normalization.
        
        Weights:
        - GitHub stars: 40%
        - Downloads: 40%
        - Active users: 20%
        
        Args:
            github_stars: Number of GitHub stars
            downloads: Number of downloads
            users: Number of active users
            
        Returns:
            Normalized score in range [0, 1]
        """
        star_score = min(github_stars / self.max_stars, 1.0) if self.max_stars > 0 else 0.0
        download_score = min(downloads / self.max_downloads, 1.0) if self.max_downloads > 0 else 0.0
        user_score = min(users / self.max_users, 1.0) if self.max_users > 0 else 0.0
        
        score = (star_score * 0.4 + download_score * 0.4 + user_score * 0.2)
        logger.debug(f"Popularity: stars={github_stars}, downloads={downloads}, users={users} -> {score}")
        return max(0.0, min(1.0, score))
    
    def compute_optimization(
        self,
        latency_ms: float = 0,
        cost_per_request: float = 0,
        doc_quality: float = 0.5
    ) -> float:
        """
        Compute optimization score.
        
        Weights:
        - Latency: 40%
        - Cost: 30%
        - Documentation quality: 30%
        
        Args:
            latency_ms: Response time in milliseconds
            cost_per_request: Cost per API request
            doc_quality: Documentation quality score (0-1)
            
        Returns:
            Normalized score in range [0, 1]
        """
        # Lower latency is better
        latency_score = max(0.0, 1.0 - (latency_ms / self.max_latency_ms)) if latency_ms >= 0 else 0.5
        
        # Lower cost is better
        cost_score = max(0.0, 1.0 - (cost_per_request / self.max_cost)) if cost_per_request >= 0 else 0.5
        
        # Higher doc quality is better
        doc_score = max(0.0, min(1.0, doc_quality))
        
        score = (latency_score * 0.4 + cost_score * 0.3 + doc_score * 0.3)
        logger.debug(f"Optimization: latency={latency_ms}ms, cost={cost_per_request}, doc={doc_quality} -> {score}")
        return max(0.0, min(1.0, score))
    
    def compute_freshness(
        self,
        last_updated: datetime,
        health_status: str = "unknown"
    ) -> float:
        """
        Compute freshness score.
        
        Weights:
        - Recency: 60%
        - Health status: 40%
        
        Args:
            last_updated: Last update timestamp
            health_status: Current health status (healthy, degraded, down, unknown)
            
        Returns:
            Normalized score in range [0, 1]
        """
        # Calculate days since last update
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        
        days_old = (datetime.utcnow() - last_updated).days
        recency_score = max(0.0, 1.0 - (days_old / 365.0))
        
        # Health status scores
        health_scores = {
            "healthy": 1.0,
            "degraded": 0.5,
            "down": 0.0,
            "unknown": 0.3
        }
        health_score = health_scores.get(health_status.lower(), 0.3)
        
        score = (recency_score * 0.6 + health_score * 0.4)
        logger.debug(f"Freshness: days_old={days_old}, health={health_status} -> {score}")
        return max(0.0, min(1.0, score))
    
    def compute_score(
        self,
        semantic_relevance: float,
        popularity: float,
        optimization: float,
        freshness: float
    ) -> float:
        """
        Compute final weighted composite score.
        
        Weights:
        - Semantic relevance: 40%
        - Popularity: 30%
        - Optimization: 20%
        - Freshness: 10%
        
        Args:
            semantic_relevance: Semantic relevance score (0-1)
            popularity: Popularity score (0-1)
            optimization: Optimization score (0-1)
            freshness: Freshness score (0-1)
            
        Returns:
            Final composite score in range [0, 1]
        """
        # Validate all inputs are in range [0, 1]
        scores = {
            'semantic_relevance': semantic_relevance,
            'popularity': popularity,
            'optimization': optimization,
            'freshness': freshness
        }
        
        for name, score in scores.items():
            if not (0.0 <= score <= 1.0):
                logger.warning(f"{name} score {score} out of range [0, 1], clamping")
                scores[name] = max(0.0, min(1.0, score))
        
        final_score = (
            scores['semantic_relevance'] * 0.4 +
            scores['popularity'] * 0.3 +
            scores['optimization'] * 0.2 +
            scores['freshness'] * 0.1
        )
        
        logger.debug(f"Final score: {final_score}")
        return max(0.0, min(1.0, final_score))
