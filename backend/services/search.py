"""Search service for DevStore"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

# Import clients and services
try:
    from clients.bedrock import BedrockClient
    from clients.opensearch import OpenSearchClient
    from services.ranking import RankingService
except ImportError:
    logger.warning("Some clients not available, using mock implementations")


class SearchService:
    """Service for semantic search using Bedrock and OpenSearch"""
    
    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        opensearch_client: Optional[OpenSearchClient] = None,
        ranking_service: Optional[RankingService] = None
    ):
        self.bedrock = bedrock_client or BedrockClient()
        self.opensearch = opensearch_client or OpenSearchClient()
        self.ranking = ranking_service or RankingService()
        self._embedding_cache = {}
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with caching"""
        if text in self._embedding_cache:
            logger.debug(f"Using cached embedding for: {text[:50]}...")
            return self._embedding_cache[text]
        
        embedding = self.bedrock.generate_embedding(text)
        self._embedding_cache[text] = embedding
        return embedding
    
    def extract_intent(self, query: str) -> Dict[str, Any]:
        """Extract search intent using fallback logic (Claude not available in this region)"""
        # Return default intent based on query keywords
        query_lower = query.lower()
        
        resource_types = ["API", "Model", "Dataset"]
        if "api" in query_lower:
            resource_types = ["API"]
        elif "model" in query_lower:
            resource_types = ["Model"]
        elif "dataset" in query_lower:
            resource_types = ["Dataset"]
        
        pricing = "both"
        if "free" in query_lower:
            pricing = "free"
        elif "paid" in query_lower:
            pricing = "paid"
        
        intent = {
            "resource_types": resource_types,
            "pricing_preference": pricing,
            "key_terms": query.split()
        }
        
        logger.info(f"Extracted intent (keyword-based): {intent}")
        return intent
    
    def vector_search(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        k: int = 10,
        size: int = 20
    ) -> List[Dict[str, Any]]:
        """Perform vector search in OpenSearch"""
        try:
            results = self.opensearch.knn_search(
                query_vector=query_embedding,
                k=k,
                filters=filters,
                size=size
            )
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def rank_results(
        self,
        results: List[Dict[str, Any]],
        query_embedding: List[float]
    ) -> List[Dict[str, Any]]:
        """Rank search results using RankingService"""
        ranked = []
        
        for result in results:
            doc = result['document']
            
            # Compute semantic relevance (cosine similarity from OpenSearch score)
            semantic_relevance = result.get('score', 0.5)
            
            # Compute other scores
            popularity = self.ranking.compute_popularity(
                github_stars=doc.get('github_stars', 0),
                downloads=doc.get('downloads', 0),
                users=doc.get('users', 0)
            )
            
            optimization = self.ranking.compute_optimization(
                latency_ms=doc.get('latency_ms', 0),
                cost_per_request=doc.get('cost_per_request', 0),
                doc_quality=doc.get('documentation_quality', 0.5)
            )
            
            from datetime import datetime
            last_updated = datetime.fromisoformat(doc.get('last_updated', datetime.utcnow().isoformat()))
            freshness = self.ranking.compute_freshness(
                last_updated=last_updated,
                health_status=doc.get('health_status', 'unknown')
            )
            
            # Compute final score
            final_score = self.ranking.compute_score(
                semantic_relevance=semantic_relevance,
                popularity=popularity,
                optimization=optimization,
                freshness=freshness
            )
            
            ranked.append({
                **doc,
                'id': result['id'],
                'score': final_score,
                'score_breakdown': {
                    'semantic_relevance': semantic_relevance,
                    'popularity': popularity,
                    'optimization': optimization,
                    'freshness': freshness
                }
            })
        
        # Sort by final score
        ranked.sort(key=lambda x: x['score'], reverse=True)
        return ranked
    
    def search(
        self,
        query: str,
        pricing_filter: Optional[List[str]] = None,
        resource_types: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Main search method orchestrating all steps
        
        Args:
            query: Natural language search query
            pricing_filter: Filter by pricing type (free, paid)
            resource_types: Filter by resource types (API, Model, Dataset)
            sources: Filter by data source (github, huggingface, kaggle)
            limit: Maximum number of results
            
        Returns:
            Search results with metadata
        """
        logger.info(f"Searching for: {query}")
        
        # Step 1: Extract intent
        intent = self.extract_intent(query)
        
        # Step 2: Generate embedding
        query_embedding = self.generate_embedding(query)
        
        # Step 3: Build filters
        filters = {}
        if pricing_filter:
            filters['pricing_type'] = pricing_filter
        elif intent.get('pricing_preference') != 'both':
            filters['pricing_type'] = [intent['pricing_preference']]
        
        if resource_types:
            filters['resource_type'] = resource_types
        elif intent.get('resource_types'):
            filters['resource_type'] = intent['resource_types']
        
        if sources:
            filters['source'] = sources
        
        # Step 4: Vector search
        results = self.vector_search(
            query_embedding=query_embedding,
            filters=filters,
            k=limit,
            size=limit
        )
        
        # Step 5: Rank results
        ranked_results = self.rank_results(results, query_embedding)
        
        # Step 6: Group by type
        grouped = {
            'API': [],
            'Model': [],
            'Dataset': []
        }
        
        for result in ranked_results[:limit]:
            resource_type = result.get('resource_type', 'API')
            if resource_type in grouped:
                grouped[resource_type].append(result)
        
        return {
            'query': query,
            'intent': intent,
            'results': ranked_results[:limit],
            'grouped_results': grouped,
            'total': len(ranked_results)
        }
    
    def get_mock_results(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Generate mock search results for testing"""
        mock_data = [
            {
                'id': '1',
                'name': 'OpenAI GPT-4 API',
                'description': 'Advanced language model API for natural language processing',
                'resource_type': 'API',
                'pricing_type': 'paid',
                'score': 0.95,
                'github_stars': 50000,
                'downloads': 1000000,
                'last_updated': datetime.utcnow().isoformat(),
                'health_status': 'healthy'
            },
            {
                'id': '2',
                'name': 'Hugging Face Transformers',
                'description': 'State-of-the-art machine learning models',
                'resource_type': 'Model',
                'pricing_type': 'free',
                'score': 0.92,
                'github_stars': 75000,
                'downloads': 2500000,
                'last_updated': datetime.utcnow().isoformat(),
                'health_status': 'healthy'
            },
            {
                'id': '3',
                'name': 'Common Crawl Dataset',
                'description': 'Petabyte-scale web crawl data',
                'resource_type': 'Dataset',
                'pricing_type': 'free',
                'score': 0.88,
                'downloads': 500000,
                'last_updated': datetime.utcnow().isoformat(),
                'health_status': 'healthy'
            }
        ]
        return mock_data[:limit]
