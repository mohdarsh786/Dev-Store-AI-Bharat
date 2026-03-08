"""
Resources API Router

Serves resources from PostgreSQL database or falls back to JSON files
"""
from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
import json
import os
import logging
from pathlib import Path

router = APIRouter(prefix="/api/resources", tags=["resources"])
logger = logging.getLogger(__name__)

# Check if database is configured
DATABASE_CONFIGURED = bool(os.getenv("DATABASE_URL"))

# Try to import repository
repository = None
if DATABASE_CONFIGURED:
    try:
        from ingestion.repository import IngestionRepository
        from clients.database import DatabaseClient
        
        db_client = DatabaseClient()
        repository = IngestionRepository(db_client)
        logger.info("✓ Using PostgreSQL database for resources")
    except Exception as e:
        logger.warning(f"Failed to initialize database: {e}")
        logger.info("Falling back to JSON files")
        repository = None
else:
    logger.info("Database not configured - using JSON files")

# Load data from JSON files (fallback)
DATA_DIR = Path(__file__).parent.parent / "ingestion" / "output"

def load_json_file(filename: str):
    """Load JSON file"""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Cache loaded data
_models_cache = None
_hf_datasets_cache = None
_kaggle_datasets_cache = None
_github_cache = None

def get_models_from_json():
    """Get all models from JSON (cached)"""
    global _models_cache
    if _models_cache is None:
        _models_cache = load_json_file("models.json")
    return _models_cache

def get_hf_datasets_from_json():
    """Get HuggingFace datasets from JSON (cached)"""
    global _hf_datasets_cache
    if _hf_datasets_cache is None:
        _hf_datasets_cache = load_json_file("huggingface_datasets.json")
    return _hf_datasets_cache

def get_kaggle_datasets_from_json():
    """Get Kaggle datasets from JSON (cached)"""
    global _kaggle_datasets_cache
    if _kaggle_datasets_cache is None:
        _kaggle_datasets_cache = load_json_file("kaggle_datasets.json")
    return _kaggle_datasets_cache

def get_github_repos_from_json():
    """Get GitHub repositories from JSON (cached)"""
    global _github_cache
    if _github_cache is None:
        _github_cache = load_json_file("github_resources.json")
    return _github_cache

def get_all_resources_from_json():
    """Get all resources from JSON files"""
    return (get_models_from_json() + get_hf_datasets_from_json() + 
            get_kaggle_datasets_from_json() + get_github_repos_from_json())

def get_all_resources():
    """Get all resources from database or JSON files"""
    if repository:
        try:
            # Get from database
            resources = repository.list_all_resources()
            logger.debug(f"Fetched {len(resources)} resources from database")
            return resources
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            logger.info("Falling back to JSON files")
    
    # Fallback to JSON files
    return get_all_resources_from_json()


@router.get("/search")
async def search_resources(
    q: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    source: Optional[str] = Query(None, description="Filter by source"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Search resources
    
    Simple text search across name, description, and tags
    """
    all_resources = get_all_resources()
    
    # Filter by query
    query_lower = q.lower()
    filtered = [
        r for r in all_resources
        if (query_lower in (r.get('name') or '').lower() or
            query_lower in (r.get('description') or '').lower() or
            any(query_lower in tag.lower() for tag in r.get('tags', [])))
    ]
    
    # Filter by category (case-insensitive, check both 'category' and 'resource_type' fields)
    if category:
        category_lower = category.lower()
        filtered = [
            r for r in filtered 
            if (r.get('category', '').lower() == category_lower or 
                r.get('resource_type', '').lower() == category_lower)
        ]
    
    # Filter by source
    if source:
        filtered = [r for r in filtered if r.get('source') == source]
    
    # Sort by popularity (stars + downloads)
    filtered.sort(
        key=lambda r: r.get('stars', 0) + r.get('downloads', 0),
        reverse=True
    )
    
    # Paginate
    total = len(filtered)
    results = filtered[offset:offset + limit]
    
    return {
        "query": q,
        "total": total,
        "limit": limit,
        "offset": offset,
        "results": results
    }


@router.get("/trending")
async def get_trending(
    category: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100)
):
    """Get trending resources"""
    all_resources = get_all_resources()
    
    # Filter by category (case-insensitive, check both 'category' and 'resource_type' fields)
    if category:
        category_lower = category.lower()
        all_resources = [
            r for r in all_resources 
            if (r.get('category', '').lower() == category_lower or 
                r.get('resource_type', '').lower() == category_lower)
        ]
    
    # Sort by stars + downloads
    all_resources.sort(
        key=lambda r: r.get('stars', 0) + r.get('downloads', 0),
        reverse=True
    )
    
    return {
        "category": category,
        "total": len(all_resources),
        "results": all_resources[:limit]
    }


@router.get("/categories")
async def get_categories():
    """Get available categories with counts"""
    all_resources = get_all_resources()
    
    categories = {}
    for resource in all_resources:
        # Check both 'category' and 'resource_type' fields
        cat = resource.get('category') or resource.get('resource_type', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "categories": [
            {"name": cat, "count": count}
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)
        ]
    }


@router.get("/sources")
async def get_sources():
    """Get available sources with counts"""
    all_resources = get_all_resources()
    
    sources = {}
    for resource in all_resources:
        src = resource.get('source', 'unknown')
        sources[src] = sources.get(src, 0) + 1
    
    return {
        "sources": [
            {"name": src, "count": count}
            for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)
        ]
    }


@router.get("/stats")
async def get_stats():
    """Get overall statistics"""
    if repository:
        try:
            # Get stats from database
            stats = repository.get_resource_stats()
            return stats
        except Exception as e:
            logger.error(f"Database stats query failed: {e}")
            logger.info("Falling back to JSON files")
    
    # Fallback to JSON files
    models = get_models_from_json()
    hf_datasets = get_hf_datasets_from_json()
    kaggle_datasets = get_kaggle_datasets_from_json()
    github = get_github_repos_from_json()
    
    return {
        "total_resources": len(models) + len(hf_datasets) + len(kaggle_datasets) + len(github),
        "models": len(models),
        "datasets": len(hf_datasets) + len(kaggle_datasets),
        "repositories": len(github),
        "by_source": {
            "huggingface": len([r for r in models if r.get('source') == 'huggingface']) + len(hf_datasets),
            "openrouter": len([r for r in models if r.get('source') == 'openrouter']),
            "github": len(github),
            "kaggle": len(kaggle_datasets)
        },
        "source": "json"
    }


@router.post("/refresh")
async def refresh_cache():
    """Refresh data cache (reload JSON files)"""
    global _models_cache, _hf_datasets_cache, _kaggle_datasets_cache, _github_cache
    
    _models_cache = None
    _hf_datasets_cache = None
    _kaggle_datasets_cache = None
    _github_cache = None
    
    # Reload
    get_models()
    get_hf_datasets()
    get_kaggle_datasets()
    get_github_repos()
    
    return {"message": "Cache refreshed successfully"}
