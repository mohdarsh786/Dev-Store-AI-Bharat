"""
Production-ready vector store with hybrid search (Vector + BM25).

This module provides:
- Hybrid search combining semantic similarity and keyword matching
- Connection management with health checks
- Query optimization and result ranking
- Error handling for timeouts and throttling
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Advanced vector store with hybrid search capabilities.
    
    Features:
    - KNN vector search for semantic similarity
    - BM25 keyword search for exact matches
    - Hybrid ranking combining both approaches
    - Connection health monitoring
    - Graceful error handling
    """
    
    def __init__(
        self,
        opensearch_client,
        bedrock_client,
        index_name: str = "devstore_resources"
    ):
        """
        Initialize vector store.
        
        Args:
            opensearch_client: OpenSearchClient instance
            bedrock_client: BedrockClient for query embeddings
            index_name: OpenSearch index name
        """
        self.opensearch = opensearch_client
        self.bedrock = bedrock_client
        self.index_name = index_name
        
    def ensure_index_exists(self) -> bool:
        """
        Check if index exists and create if needed.
        
        Returns:
            True if index exists or was created
        """
        try:
            if self.opensearch.index_exists(self.index_name):
                logger.info(f"Index '{self.index_name}' exists")
                return True
            
            logger.warning(f"Index '{self.index_name}' does not exist, creating...")
            return self.opensearch.create_knn_index(
                index_name=self.index_name,
                vector_dimension=1024,  # Titan v2 dimension
                vector_field="embedding"
            )
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on vector store.
        
        Returns:
            Health status dict
        """
        try:
            os_health = self.opensearch.health_check()
            
            # Check if index exists
            index_exists = self.opensearch.index_exists(self.index_name)
            
            # Get document count
            doc_count = 0
            if index_exists:
                try:
                    response = self.opensearch._client.count(index=self.index_name)
                    doc_count = response.get('count', 0)
                except:
                    pass
            
            return {
                'status': 'healthy' if os_health['status'] == 'healthy' and index_exists else 'unhealthy',
                'opensearch': os_health,
                'index_exists': index_exists,
                'document_count': doc_count,
                'index_name': self.index_name
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def hybrid_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 10,
        size: int = 20,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Search query
            filters: Optional filters (resource_type, pricing_type, etc.)
            k: Number of nearest neighbors for KNN
            size: Maximum results to return
            alpha: Weight for vector search (0-1), (1-alpha) for keyword search
            
        Returns:
            List of search results with hybrid scores
        """
        try:
            # Generate query embedding
            query_embedding = self.bedrock.generate_embedding(query)
            
            # Build hybrid query
            query_body = self._build_hybrid_query(
                query=query,
                query_embedding=query_embedding,
                filters=filters,
                k=k,
                alpha=alpha
            )
            
            # Execute search
            response = self.opensearch._client.search(
                index=self.index_name,
                body=query_body,
                size=size
            )
            
            # Process results
            results = self._process_search_results(response, alpha)
            
            logger.info(f"Hybrid search returned {len(results)} results for query: {query[:50]}")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def _build_hybrid_query(
        self,
        query: str,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]],
        k: int,
        alpha: float
    ) -> Dict[str, Any]:
        """
        Build OpenSearch query combining KNN and BM25.
        
        The query uses:
        - KNN for semantic similarity (weighted by alpha)
        - Multi-match for keyword relevance (weighted by 1-alpha)
        - Filters for resource_type, pricing_type, etc.
        """
        # KNN query for vector search
        knn_query = {
            "embedding": {
                "vector": query_embedding,
                "k": k
            }
        }
        
        # BM25 query for keyword search
        keyword_query = {
            "multi_match": {
                "query": query,
                "fields": ["name^3", "description^2", "tags^1.5", "author"],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }
        
        # Combine queries with weights
        query_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": knn_query,
                            "boost": alpha
                        },
                        {
                            **keyword_query,
                            "boost": 1 - alpha
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }
        
        # Add filters
        if filters:
            filter_clauses = []
            for field, value in filters.items():
                if isinstance(value, list):
                    filter_clauses.append({"terms": {field: value}})
                else:
                    filter_clauses.append({"term": {field: value}})
            
            if filter_clauses:
                query_body["query"]["bool"]["filter"] = filter_clauses
        
        return query_body
    
    def _process_search_results(
        self,
        response: Dict[str, Any],
        alpha: float
    ) -> List[Dict[str, Any]]:
        """Process and normalize search results"""
        hits = response.get('hits', {}).get('hits', [])
        results = []
        
        for hit in hits:
            result = {
                'id': hit['_id'],
                'score': hit['_score'],
                'document': hit['_source'],
                'search_type': 'hybrid',
                'vector_weight': alpha,
                'keyword_weight': 1 - alpha
            }
            results.append(result)
        
        return results
    
    def vector_only_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 10,
        size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform pure vector search (semantic only).
        
        Args:
            query: Search query
            filters: Optional filters
            k: Number of nearest neighbors
            size: Maximum results
            
        Returns:
            List of search results
        """
        try:
            query_embedding = self.bedrock.generate_embedding(query)
            
            results = self.opensearch.knn_search(
                query_vector=query_embedding,
                k=k,
                filters=filters,
                index_name=self.index_name,
                size=size
            )
            
            logger.info(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def keyword_only_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform pure keyword search (BM25 only).
        
        Args:
            query: Search query
            filters: Optional filters
            size: Maximum results
            
        Returns:
            List of search results
        """
        try:
            query_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["name^3", "description^2", "tags^1.5"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ]
                    }
                }
            }
            
            # Add filters
            if filters:
                filter_clauses = []
                for field, value in filters.items():
                    if isinstance(value, list):
                        filter_clauses.append({"terms": {field: value}})
                    else:
                        filter_clauses.append({"term": {field: value}})
                
                if filter_clauses:
                    query_body["query"]["bool"]["filter"] = filter_clauses
            
            response = self.opensearch._client.search(
                index=self.index_name,
                body=query_body,
                size=size
            )
            
            # Process results
            hits = response.get('hits', {}).get('hits', [])
            results = []
            for hit in hits:
                results.append({
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'document': hit['_source'],
                    'search_type': 'keyword'
                })
            
            logger.info(f"Keyword search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
