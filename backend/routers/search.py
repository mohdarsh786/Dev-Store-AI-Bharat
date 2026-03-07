"""Search router for DevStore API."""

from __future__ import annotations

from datetime import datetime
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["search"])

# Check if AWS services are configured
AWS_CONFIGURED = all([
    os.getenv("AWS_REGION"),
    os.getenv("OPENSEARCH_HOST"),
    os.getenv("DATABASE_URL"),
])

search_service = None
if AWS_CONFIGURED:
    try:
        from services.search import SearchService

        search_service = SearchService()
        logger.info("AWS services configured - using real search")
    except Exception as e:
        logger.warning(f"Failed to initialize search service: {e}")
        logger.info("Falling back to mock data")
        search_service = None
else:
    logger.info("AWS not configured - using mock data")

try:
    from ingestion.repository import IngestionRepository
except Exception:
    IngestionRepository = None


class SearchRequest(BaseModel):
    query: str
    pricing_filter: Optional[List[str]] = None
    resource_types: Optional[List[str]] = None
    limit: int = 20


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    grouped_results: Optional[Dict[str, List[Dict[str, Any]]]] = None
    total: int
    intent: Optional[Dict[str, Any]] = None
    source: str = "mock"


def get_mock_results(
    query: str,
    pricing_filter: Optional[List[str]] = None,
    resource_types: Optional[List[str]] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    all_results = [
        {
            "id": "1",
            "name": "OpenAI GPT-4 API",
            "description": "Advanced language model API for natural language processing, text generation, and conversational AI",
            "resource_type": "API",
            "pricing_type": "paid",
            "score": 0.95,
            "rank": 1,
            "github_stars": 50000,
            "downloads": 1000000,
            "documentation_url": "https://platform.openai.com/docs",
            "health_status": "healthy",
            "last_updated": datetime.utcnow().isoformat(),
        },
        {
            "id": "2",
            "name": "Hugging Face Transformers",
            "description": "State-of-the-art machine learning models for NLP tasks including classification, translation, and generation",
            "resource_type": "Model",
            "pricing_type": "free",
            "score": 0.92,
            "rank": 2,
            "github_stars": 75000,
            "downloads": 2500000,
            "documentation_url": "https://huggingface.co/docs",
            "health_status": "healthy",
            "last_updated": datetime.utcnow().isoformat(),
        },
        {
            "id": "3",
            "name": "Common Crawl Dataset",
            "description": "Petabyte-scale web crawl data for training large language models and research",
            "resource_type": "Dataset",
            "pricing_type": "free",
            "score": 0.88,
            "rank": 5,
            "downloads": 500000,
            "documentation_url": "https://commoncrawl.org",
            "health_status": "healthy",
            "last_updated": datetime.utcnow().isoformat(),
        },
        {
            "id": "4",
            "name": "Anthropic Claude API",
            "description": "Constitutional AI assistant with advanced reasoning capabilities and safety features",
            "resource_type": "API",
            "pricing_type": "paid",
            "score": 0.87,
            "rank": 3,
            "github_stars": 30000,
            "downloads": 500000,
            "documentation_url": "https://docs.anthropic.com",
            "health_status": "healthy",
            "last_updated": datetime.utcnow().isoformat(),
        },
        {
            "id": "5",
            "name": "Stable Diffusion Models",
            "description": "Open-source text-to-image generation models for creating high-quality images",
            "resource_type": "Model",
            "pricing_type": "free",
            "score": 0.85,
            "rank": 4,
            "github_stars": 60000,
            "downloads": 1500000,
            "documentation_url": "https://stability.ai/docs",
            "health_status": "healthy",
            "last_updated": datetime.utcnow().isoformat(),
        },
    ]

    filtered = all_results
    if pricing_filter:
        filtered = [item for item in filtered if item["pricing_type"] in pricing_filter]
    if resource_types and "All" not in resource_types:
        filtered = [item for item in filtered if item["resource_type"] in resource_types]
    if query:
        query_lower = query.lower()
        filtered = [
            item
            for item in filtered
            if query_lower in item["name"].lower() or query_lower in item["description"].lower()
        ]
    return filtered[:limit]


@router.post("/search")
async def search(request: SearchRequest) -> SearchResponse:
    try:
        if search_service is not None:
            try:
                result = search_service.search(
                    query=request.query,
                    pricing_filter=request.pricing_filter,
                    resource_types=request.resource_types,
                    limit=request.limit,
                )
                return SearchResponse(
                    query=result["query"],
                    results=result["results"],
                    grouped_results=result.get("grouped_results"),
                    total=result["total"],
                    intent=result.get("intent"),
                    source="aws",
                )
            except Exception as e:
                logger.warning(f"AWS search failed, falling back to mock: {e}")

        results = get_mock_results(
            query=request.query,
            pricing_filter=request.pricing_filter,
            resource_types=request.resource_types,
            limit=request.limit,
        )
        grouped = {"API": [], "Model": [], "Dataset": []}
        for result in results:
            resource_type = result.get("resource_type", "API")
            if resource_type in grouped:
                grouped[resource_type].append(result)
        intent = {
            "resource_types": request.resource_types or ["API", "Model", "Dataset"],
            "pricing_preference": request.pricing_filter[0] if request.pricing_filter else "both",
            "key_terms": request.query.split() if request.query else [],
        }
        return SearchResponse(
            query=request.query,
            results=results,
            grouped_results=grouped,
            total=len(results),
            intent=intent,
            source="mock",
        )
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/trending")
async def trending(resource_type: Optional[str] = None, limit: int = 40) -> SearchResponse:
    try:
        if IngestionRepository is not None and os.getenv("DATABASE_URL"):
            try:
                repository = IngestionRepository()
                normalized_type = None if resource_type in (None, "All") else resource_type.lower()
                results = repository.list_trending_resources(normalized_type, limit)
                if results:
                    for item in results:
                        item["score"] = item.get("rank_score", 0.0)
                        item["rank"] = item.get("category_rank")
                    return SearchResponse(
                        query="trending",
                        results=results,
                        total=len(results),
                        source="database",
                    )
            except Exception as e:
                logger.warning(f"Database trending lookup failed, falling back to mock: {e}")

        results = get_mock_results(query="", limit=100)
        if resource_type and resource_type != "All":
            results = [item for item in results if item["resource_type"] == resource_type]
        results.sort(key=lambda item: item.get("rank", 999))
        final_results = results[:limit]
        return SearchResponse(
            query="trending",
            results=final_results,
            total=len(final_results),
            source="mock",
        )
    except Exception as e:
        logger.error(f"Trending failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
