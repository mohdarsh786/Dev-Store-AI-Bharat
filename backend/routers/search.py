"""Search router for DevStore API"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])

# Check if AWS services are configured
AWS_CONFIGURED = all([
    os.getenv('AWS_REGION'),
    os.getenv('OPENSEARCH_HOST'),
    os.getenv('DATABASE_URL')
])

# Try to import search service if AWS is configured
search_service = None
if AWS_CONFIGURED:
    try:
        from services.search import SearchService
        search_service = SearchService()
        logger.info("✅ AWS services configured - using real search")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize search service: {e}")
        logger.info("📦 Falling back to mock data")
        search_service = None
else:
    logger.info("📦 AWS not configured - using mock data (set AWS_REGION, OPENSEARCH_HOST, DATABASE_URL to enable real search)")


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
    source: str = "mock"  # "mock" or "aws"


def get_mock_results(query: str, pricing_filter: Optional[List[str]] = None, resource_types: Optional[List[str]] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Generate mock search results"""
    all_results = [
        {
            'id': '1',
            'name': 'OpenAI GPT-4 API',
            'description': 'Advanced language model API for natural language processing, text generation, and conversational AI',
            'resource_type': 'API',
            'pricing_type': 'paid',
            'score': 0.95,
            'rank': 1,
            'github_stars': 50000,
            'downloads': 1000000,
            'documentation_url': 'https://platform.openai.com/docs',
            'health_status': 'healthy',
            'last_updated': datetime.utcnow().isoformat()
        },
        {
            'id': '2',
            'name': 'Hugging Face Transformers',
            'description': 'State-of-the-art machine learning models for NLP tasks including classification, translation, and generation',
            'resource_type': 'Model',
            'pricing_type': 'free',
            'score': 0.92,
            'rank': 2,
            'github_stars': 75000,
            'downloads': 2500000,
            'documentation_url': 'https://huggingface.co/docs',
            'health_status': 'healthy',
            'last_updated': datetime.utcnow().isoformat()
        },
        {
            'id': '3',
            'name': 'Common Crawl Dataset',
            'description': 'Petabyte-scale web crawl data for training large language models and research',
            'resource_type': 'Dataset',
            'pricing_type': 'free',
            'score': 0.88,
            'rank': 5,
            'downloads': 500000,
            'documentation_url': 'https://commoncrawl.org',
            'health_status': 'healthy',
            'last_updated': datetime.utcnow().isoformat()
        },
        {
            'id': '4',
            'name': 'Anthropic Claude API',
            'description': 'Constitutional AI assistant with advanced reasoning capabilities and safety features',
            'resource_type': 'API',
            'pricing_type': 'paid',
            'score': 0.87,
            'rank': 3,
            'github_stars': 30000,
            'downloads': 500000,
            'documentation_url': 'https://docs.anthropic.com',
            'health_status': 'healthy',
            'last_updated': datetime.utcnow().isoformat()
        },
        {
            'id': '5',
            'name': 'Stable Diffusion Models',
            'description': 'Open-source text-to-image generation models for creating high-quality images',
            'resource_type': 'Model',
            'pricing_type': 'free',
            'score': 0.85,
            'rank': 4,
            'github_stars': 60000,
            'downloads': 1500000,
            'documentation_url': 'https://stability.ai/docs',
            'health_status': 'healthy',
            'last_updated': datetime.utcnow().isoformat()
        },
        {
            'id': '6',
            'name': 'ImageNet Dataset',
            'description': 'Large-scale image database for visual recognition research and training',
            'resource_type': 'Dataset',
            'pricing_type': 'free',
            'score': 0.82,
            'rank': 6,
            'downloads': 800000,
            'documentation_url': 'https://www.image-net.org',
            'health_status': 'healthy',
            'last_updated': datetime.utcnow().isoformat()
        },
        {
            'id': '7',
            'name': 'Google Gemini API',
            'description': 'Multimodal AI model for text, image, and code generation',
            'resource_type': 'API',
            'pricing_type': 'paid',
            'score': 0.90,
            'rank': 7,
            'github_stars': 40000,
            'downloads': 750000,
            'documentation_url': 'https://ai.google.dev/docs',
            'health_status': 'healthy',
            'last_updated': datetime.utcnow().isoformat()
        },
        {
            'id': '8',
            'name': 'Llama 3 70B',
            'description': 'Meta\'s open-source large language model with 70 billion parameters',
            'resource_type': 'Model',
            'pricing_type': 'free',
            'score': 0.89,
            'rank': 8,
            'github_stars': 85000,
            'downloads': 3000000,
            'documentation_url': 'https://llama.meta.com',
            'health_status': 'healthy',
            'last_updated': datetime.utcnow().isoformat()
        }
    ]
    
    # Apply filters
    filtered = all_results
    
    if pricing_filter:
        filtered = [r for r in filtered if r['pricing_type'] in pricing_filter]
    
    if resource_types and 'All' not in resource_types:
        filtered = [r for r in filtered if r['resource_type'] in resource_types]
    
    # Filter by query keywords (simple matching)
    if query:
        query_lower = query.lower()
        filtered = [r for r in filtered if 
                   query_lower in r['name'].lower() or 
                   query_lower in r['description'].lower()]
    
    return filtered[:limit]


@router.post("/search")
async def search(request: SearchRequest) -> SearchResponse:
    """
    Search for resources using AI-powered semantic search
    
    Behavior:
    - If AWS services configured (AWS_REGION, OPENSEARCH_HOST, DATABASE_URL): Uses real Bedrock + OpenSearch
    - Otherwise: Returns mock data for testing
    
    Real search flow:
    1. Extract intent from natural language query using Bedrock (Claude)
    2. Generate embedding vector using Bedrock (Titan)
    3. Perform KNN vector search in OpenSearch
    4. Rank results using composite scoring
    5. Group results by resource type
    """
    try:
        logger.info(f"🔍 Search request: query='{request.query}', filters={request.pricing_filter}, types={request.resource_types}")
        
        # Try real search if AWS is configured
        if search_service is not None:
            try:
                logger.info("🌐 Using AWS Bedrock + OpenSearch for search")
                result = search_service.search(
                    query=request.query,
                    pricing_filter=request.pricing_filter,
                    resource_types=request.resource_types,
                    limit=request.limit
                )
                
                return SearchResponse(
                    query=result['query'],
                    results=result['results'],
                    grouped_results=result.get('grouped_results'),
                    total=result['total'],
                    intent=result.get('intent'),
                    source="aws"
                )
            except Exception as e:
                logger.warning(f"⚠️ AWS search failed, falling back to mock: {e}")
        
        # Fallback to mock data
        logger.info("📦 Using mock data")
        results = get_mock_results(
            query=request.query,
            pricing_filter=request.pricing_filter,
            resource_types=request.resource_types,
            limit=request.limit
        )
        
        # Group by type
        grouped = {
            'API': [],
            'Model': [],
            'Dataset': []
        }
        
        for result in results:
            resource_type = result.get('resource_type', 'API')
            if resource_type in grouped:
                grouped[resource_type].append(result)
        
        # Mock intent
        intent = {
            'resource_types': request.resource_types or ['API', 'Model', 'Dataset'],
            'pricing_preference': request.pricing_filter[0] if request.pricing_filter else 'both',
            'key_terms': request.query.split() if request.query else []
        }
        
        return SearchResponse(
            query=request.query,
            results=results,
            grouped_results=grouped,
            total=len(results),
            intent=intent,
            source="mock"
        )
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )
@router.get("/trending")
async def trending(
    resource_type: Optional[str] = None,
    limit: int = 40
) -> SearchResponse:
    """
    Get trending resources sorted by rank (popularity)
    """
    try:
        # For now, using mock results with sorting
        results = get_mock_results(query="", limit=100)
        
        # Apply type filter if provided
        if resource_type and resource_type != "All":
            results = [r for r in results if r['resource_type'] == resource_type]
            
        # Sort by rank (ascending: Rank 1 is best)
        results.sort(key=lambda x: x.get('rank', 999))
        
        final_results = results[:limit]
        
        return SearchResponse(
            query="trending",
            results=final_results,
            total=len(final_results),
            source="mock"
        )
    except Exception as e:
        logger.error(f"❌ Trending failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
