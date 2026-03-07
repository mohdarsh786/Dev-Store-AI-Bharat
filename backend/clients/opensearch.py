"""
OpenSearch client module for DevStore

Provides OpenSearch connection with index management and KNN vector search.
"""
import time
import logging
from typing import Optional, Any, Dict, List
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import (
    ConnectionError as OpenSearchConnectionError,
    TransportError,
    NotFoundError
)

logger = logging.getLogger(__name__)


class OpenSearchClientError(Exception):
    """Raised when OpenSearch operations fail after retries"""
    pass


class OpenSearchClient:
    """
    OpenSearch client with connection management and KNN vector search.
    
    Features:
    - Connection management with retry logic
    - Index management (create, delete, exists)
    - Document indexing
    - KNN vector search for semantic similarity
    - Error handling for connection failures
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: int = None,
        use_ssl: bool = None,
        index_name: Optional[str] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0
    ):
        """
        Initialize OpenSearch client.
        
        Args:
            host: OpenSearch host (required if not using settings)
            port: OpenSearch port (defaults to 443)
            use_ssl: Whether to use SSL (defaults to True)
            index_name: Default index name (defaults to "devstore_resources")
            max_retries: Maximum number of retry attempts for failed operations
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds between retries
        """
        # Import settings only when needed to avoid initialization issues in tests
        if host is None:
            from config import settings
            host = settings.opensearch_host
            port = port or settings.opensearch_port
            use_ssl = use_ssl if use_ssl is not None else settings.opensearch_use_ssl
            index_name = index_name or settings.opensearch_index_name
        
        # Strip protocol if present in host
        if host:
            host = host.replace("https://", "").replace("http://", "")
            if host.endswith("/"):
                host = host[:-1]

        self.host = host
        self.port = port or 443
        self.use_ssl = use_ssl if use_ssl is not None else True
        self.index_name = index_name or "devstore_resources"
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        self._client: Optional[OpenSearch] = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the OpenSearch client with retry logic."""
        
        # Configure AWS SigV4 auth if applicable
        http_auth = None
        try:
            from config import settings
            # Check if using AWS OpenSearch Service
            if "amazonaws.com" in self.host and settings.aws_access_key_id and settings.aws_secret_access_key:
                try:
                    from requests_aws4auth import AWS4Auth
                    # Determine service name ('aoss' for serverless, 'es' for managed)
                    service = "aoss" if "aoss.amazonaws.com" in self.host else "es"
                    region = settings.aws_region
                    
                    http_auth = AWS4Auth(
                        settings.aws_access_key_id,
                        settings.aws_secret_access_key,
                        region,
                        service
                    )
                    logger.info(f"Configured AWS SigV4 auth for service: {service}")
                except ImportError:
                    logger.warning("requests-aws4auth not installed. AWS OpenSearch authentication may fail.")
        except Exception as e:
             logger.warning(f"Error configuring AWS auth: {str(e)}")

        for attempt in range(self.max_retries):
            try:
                self._client = OpenSearch(
                    hosts=[{'host': self.host, 'port': self.port}],
                    http_auth=http_auth,  # Use IAM auth in production
                    use_ssl=self.use_ssl,
                    verify_certs=True,
                    connection_class=RequestsHttpConnection,
                    timeout=30,
                    max_retries=1,
                    retry_on_timeout=True
                )
                
                # Test connection
                try:
                    info = self._client.info()
                    logger.info(
                        f"OpenSearch client initialized successfully "
                        f"(cluster={info.get('cluster_name')}, version={info.get('version', {}).get('number')})"
                    )
                except NotFoundError:
                    # OpenSearch Serverless does not support the root endpoint (GET /)
                    if "aoss.amazonaws.com" in self.host:
                        logger.info("OpenSearch Serverless detected (root endpoint 404 expected). Client initialized.")
                    else:
                        raise
                return
            except (OpenSearchConnectionError, TransportError) as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to initialize OpenSearch client after {self.max_retries} attempts: {e}")
                    raise OpenSearchClientError(
                        f"Could not connect to OpenSearch after {self.max_retries} attempts"
                    ) from e
                
                delay = self._calculate_backoff_delay(attempt)
                logger.warning(
                    f"OpenSearch connection attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds, capped at max_delay
        """
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)
    
    def create_index(
        self,
        index_name: Optional[str] = None,
        mapping: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create an index with optional mapping and settings.
        
        Args:
            index_name: Name of the index (defaults to self.index_name)
            mapping: Index mapping configuration
            settings: Index settings configuration
            
        Returns:
            True if index was created, False if it already exists
            
        Raises:
            OpenSearchClientError: If index creation fails
        """
        index_name = index_name or self.index_name
        
        if self.index_exists(index_name):
            logger.info(f"Index '{index_name}' already exists")
            return False
        
        body = {}
        if settings:
            body['settings'] = settings
        if mapping:
            body['mappings'] = mapping
        
        try:
            response = self._client.indices.create(index=index_name, body=body)
            logger.info(f"Index '{index_name}' created successfully")
            return response.get('acknowledged', False)
        except TransportError as e:
            logger.error(f"Failed to create index '{index_name}': {e}")
            raise OpenSearchClientError(f"Could not create index '{index_name}'") from e
    
    def create_knn_index(
        self,
        index_name: Optional[str] = None,
        vector_dimension: int = 1024,
        vector_field: str = "embedding"
    ) -> bool:
        """
        Create an index with k-NN vector search support.
        
        Args:
            index_name: Name of the index (defaults to self.index_name)
            vector_dimension: Dimension of embedding vectors (1024 for Titan v2)
            vector_field: Name of the vector field (defaults to "embedding")
            
        Returns:
            True if index was created, False if it already exists
            
        Raises:
            OpenSearchClientError: If index creation fails
        """
        settings = {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            }
        }
        
        mapping = {
            "properties": {
                vector_field: {
                    "type": "knn_vector",
                    "dimension": vector_dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 24
                        }
                    }
                },
                "name": {"type": "text"},
                "description": {"type": "text"},
                "resource_type": {"type": "keyword"},
                "pricing_type": {"type": "keyword"},
                "source": {"type": "keyword"},
                "health_status": {"type": "keyword"},
                "github_stars": {"type": "integer"},
                "downloads": {"type": "integer"},
                "last_updated": {"type": "date"}
            }
        }
        
        return self.create_index(
            index_name=index_name,
            mapping=mapping,
            settings=settings
        )
    
    def delete_index(self, index_name: Optional[str] = None) -> bool:
        """
        Delete an index.
        
        Args:
            index_name: Name of the index (defaults to self.index_name)
            
        Returns:
            True if index was deleted, False if it didn't exist
            
        Raises:
            OpenSearchClientError: If index deletion fails
        """
        index_name = index_name or self.index_name
        
        if not self.index_exists(index_name):
            logger.info(f"Index '{index_name}' does not exist")
            return False
        
        try:
            response = self._client.indices.delete(index=index_name)
            logger.info(f"Index '{index_name}' deleted successfully")
            return response.get('acknowledged', False)
        except TransportError as e:
            logger.error(f"Failed to delete index '{index_name}': {e}")
            raise OpenSearchClientError(f"Could not delete index '{index_name}'") from e
    
    def index_exists(self, index_name: Optional[str] = None) -> bool:
        """
        Check if an index exists.
        
        Args:
            index_name: Name of the index (defaults to self.index_name)
            
        Returns:
            True if index exists, False otherwise
        """
        index_name = index_name or self.index_name
        
        try:
            return self._client.indices.exists(index=index_name)
        except TransportError as e:
            logger.error(f"Failed to check if index '{index_name}' exists: {e}")
            return False
    
    def index_document(
        self,
        document: Dict[str, Any],
        doc_id: Optional[str] = None,
        index_name: Optional[str] = None,
        refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Index a document.
        
        Args:
            document: Document to index
            doc_id: Document ID (auto-generated if not provided)
            index_name: Name of the index (defaults to self.index_name)
            refresh: Whether to refresh the index immediately
            
        Returns:
            Response from OpenSearch with document ID and result
            
        Raises:
            OpenSearchClientError: If indexing fails
        """
        index_name = index_name or self.index_name
        
        try:
            response = self._client.index(
                index=index_name,
                body=document,
                id=doc_id,
                refresh=refresh
            )
            logger.debug(f"Document indexed successfully in '{index_name}' with ID {response['_id']}")
            return response
        except TransportError as e:
            logger.error(f"Failed to index document in '{index_name}': {e}")
            raise OpenSearchClientError(f"Could not index document in '{index_name}'") from e
    
    def knn_search(
        self,
        query_vector: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
        size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform KNN vector search.
        
        Args:
            query_vector: Query embedding vector
            k: Number of nearest neighbors to find
            filters: Optional filters to apply (e.g., pricing_type, health_status)
            index_name: Name of the index (defaults to self.index_name)
            size: Maximum number of results to return
            
        Returns:
            List of matching documents with scores
            
        Raises:
            OpenSearchClientError: If search fails
        """
        index_name = index_name or self.index_name
        
        # Build KNN query
        knn_query = {
            "embedding": {
                "vector": query_vector,
                "k": k
            }
        }
        
        # Build the query body
        query_body = {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {"knn": knn_query}
                    ]
                }
            }
        }
        
        # Add filters if provided
        if filters:
            filter_clauses = []
            for field, value in filters.items():
                if isinstance(value, list):
                    filter_clauses.append({"terms": {field: value}})
                else:
                    filter_clauses.append({"term": {field: value}})
            
            if filter_clauses:
                query_body["query"]["bool"]["filter"] = filter_clauses
        
        try:
            response = self._client.search(
                index=index_name,
                body=query_body
            )
            
            # Extract hits
            hits = response.get('hits', {}).get('hits', [])
            results = []
            for hit in hits:
                result = {
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'document': hit['_source']
                }
                results.append(result)
            
            logger.debug(f"KNN search returned {len(results)} results from '{index_name}'")
            return results
            
        except (TransportError, NotFoundError) as e:
            logger.error(f"KNN search failed in '{index_name}': {e}")
            raise OpenSearchClientError(f"Could not perform KNN search in '{index_name}'") from e
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the OpenSearch connection.
        
        Returns:
            Dictionary with health check results:
            {
                "status": "healthy" | "unhealthy",
                "cluster_name": str,
                "cluster_status": str,
                "response_time_ms": float,
                "error": str (if unhealthy)
            }
        """
        start_time = time.time()
        result = {
            "status": "unhealthy",
            "response_time_ms": 0.0
        }
        
        try:
            # Get cluster health
            health = self._client.cluster.health()
            info = self._client.info()
            
            response_time = (time.time() - start_time) * 1000
            result.update({
                "status": "healthy",
                "cluster_name": info.get('cluster_name'),
                "cluster_status": health.get('status'),
                "response_time_ms": round(response_time, 2)
            })
            
            logger.info(
                f"OpenSearch health check passed "
                f"(cluster_status={health.get('status')}, response_time={response_time:.2f}ms)"
            )
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"OpenSearch health check failed: {e}")
        
        return result
    
    def close(self) -> None:
        """Close the OpenSearch client connection."""
        if self._client:
            # OpenSearch client doesn't have an explicit close method
            # Connection will be closed when object is garbage collected
            logger.info("OpenSearch client closed")
            self._client = None
    
    def __enter__(self):
        """Support for context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context manager."""
        self.close()


# Global OpenSearch client instance (can be initialized when needed)
# Usage: from clients.opensearch import OpenSearchClient; os_client = OpenSearchClient()
