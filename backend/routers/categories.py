"""
Categories Router - Browse resources by category
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from uuid import UUID

router = APIRouter(
    prefix=""
)


@router.get("/categories")
async def list_categories():
    """
    List all categories with subcategories
    
    Returns categories organized by resource type (API, Model, Dataset)
    with subcategories (Top Grossing, Top Free, Top Paid, Trending, New Releases)
    """
    # TODO: Implement with database query
    return {
        "categories": [
            {
                "id": "api",
                "name": "APIs",
                "subcategories": [
                    {"id": "top-grossing", "name": "Top Grossing"},
                    {"id": "top-free", "name": "Top Free"},
                    {"id": "top-paid", "name": "Top Paid"},
                    {"id": "trending", "name": "Trending"},
                    {"id": "new-releases", "name": "New Releases"}
                ]
            },
            {
                "id": "model",
                "name": "Models",
                "subcategories": [
                    {"id": "top-grossing", "name": "Top Grossing"},
                    {"id": "top-free", "name": "Top Free"},
                    {"id": "top-paid", "name": "Top Paid"},
                    {"id": "trending", "name": "Trending"},
                    {"id": "new-releases", "name": "New Releases"}
                ]
            },
            {
                "id": "dataset",
                "name": "Datasets",
                "subcategories": [
                    {"id": "top-grossing", "name": "Top Grossing"},
                    {"id": "top-free", "name": "Top Free"},
                    {"id": "top-paid", "name": "Top Paid"},
                    {"id": "trending", "name": "Trending"},
                    {"id": "new-releases", "name": "New Releases"}
                ]
            }
        ]
    }


@router.get("/categories/{category_id}/resources")
async def get_category_resources(
    category_id: str,
    subcategory: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """
    Get resources in a specific category/subcategory
    
    Args:
        category_id: Category identifier (api, model, dataset)
        subcategory: Subcategory filter (top-grossing, top-free, etc.)
        page: Page number
        page_size: Items per page
    """
    # TODO: Implement with database query and caching
    return {
        "category_id": category_id,
        "subcategory": subcategory,
        "resources": [],
        "total": 0,
        "page": page,
        "page_size": page_size
    }
