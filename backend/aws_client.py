"""
Unified AWS Client for Dev-Store

Provides authenticated access to:
- Amazon OpenSearch Serverless (with AWS SigV4 signing)
- AWS Bedrock (Claude 3.5 Sonnet + Titan Embeddings v2)

Features:
- Dynamic IAM credential resolution
- Request signing for OpenSearch
- Connection validation
- Robust error handling
"""
import json
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from requests_aws4auth import AWS4Auth

logger = logging.getLogger(__name__)


class AWSClientError(Exception):
    """Raised when AWS operations fail"""
    pass


class AWSClient:
    """
    Unified AWS client with IAM authentication for OpenSearch and Bedrock.
    
    This client handles:
    - Automatic credential resolution (IAM role, access keys, or environment)
    - Request signing for OpenSearch Serverless/Service
    - Bedrock model invocation (Claude 3.5 Sonnet, Titan v2)
    - Connection health checks
    """
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        opensearch_host: Optional[str] = None,
        opensearch_index: str = "devstore_resources",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None
    ):
        """
        Initialize AWS client with IAM authentication.
        
        Args:
            region_name: AWS region
            opensearch_host: OpenSearch endpoint (without https://)
            opensearch_index: Index name
            aws_access_key_id: Optional access key (uses IAM role if not provided)
            aws_secret_access_key: Optional secret key
        """
        self.region_name = region_name
        self.opensearch_host = opensearch_host
        self.opensearch_index = opensearch_index
        
        # Initialize boto3 session with credentials
        session_kwargs = {'region_name': region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key
            logger.info("Using explicit AWS credentials")
        else:
            logger.info("Using default AWS credentials (IAM role or environment)")
        
        self.session = boto3.Session(**session_kwargs)
        self.credentials = self.session.get_credentials()
        
        if not self.credentials:
            raise AWSClientError("No AWS credentials found. Configure IAM user or role.")
        
        # Initialize clients
        self.bedrock_runtime = self.session.client('bedrock-runtime')
        self.opensearch_client = None
        
        if opensearch_host:
            self._initialize_opensearch()
        
        logger.info(f"AWSClient initialized (region={region_name})")
    
    def _initialize_opensearch(self) -> None:
        """Initialize OpenSearch client with AWS SigV4 authentication."""
        try:
            # Clean host URL
            host = self.opensearch_host.replace("https://", "").replace("http://", "")
            if host.endswith("/"):
                host = host[:-1]
            
            # Determine service name (aoss for Serverless, es for managed)
            service = "aoss" if "aoss.amazonaws.com" in host else "es"
            
            # Create AWS4Auth signer using current credentials
            awsauth = AWS4Auth(
                self.credentials.access_key,
                self.credentials.secret_key,
                self.region_name,
                service,
                session_token=self.credentials.token  # For temporary credentials
            )
            
            # Initialize OpenSearch client with signed requests
            self.opensearch_client = OpenSearch(
                hosts=[{'host': host, 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )
            
            logger.info(f"OpenSearch client initialized (host={host}, service={service})")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenSearch client: {e}")
            raise AWSClientError(f"OpenSearch initialization failed: {e}") from e
    
    def check_connection(self) -> Dict[str, Any]:
        """
        Validate connections to OpenSearch and Bedrock.
        
        Returns:
            Dict with connection status for each service
        """
        results = {
            "opensearch": {"status": "not_configured"},
            "bedrock": {"status": "unknown"}
        }
        
        # Check OpenSearch
        if self.opensearch_client:
            try:
                # For Serverless, try to check if index exists (root endpoint not supported)
                if "aoss.amazonaws.com" in self.opensearch_host:
                    # Serverless doesn't support cluster health, check index instead
                    exists = self.opensearch_client.indices.exists(index=self.opensearch_index)
                    results["opensearch"] = {
                        "status": "healthy",
                        "service": "serverless",
                        "index_exists": exists,
                        "message": f"✅ OpenSearch Serverless connected (index: {self.opensearch_index})"
                    }
                else:
                    # Managed OpenSearch supports cluster health
                    health = self.opensearch_client.cluster.health()
                    results["opensearch"] = {
                        "status": "healthy",
                        "service": "managed",
                        "cluster_status": health.get('status'),
                        "message": f"✅ OpenSearch connected (cluster: {health.get('cluster_name')})"
                    }
                
                logger.info(results["opensearch"]["message"])
                
            except Exception as e:
                results["opensearch"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "message": f"❌ OpenSearch connection failed: {e}"
                }
                logger.error(results["opensearch"]["message"])
        
        # Check Bedrock
        try:
            # Test with a simple embedding generation
            test_response = self.bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({
                    "inputText": "connection test",
                    "dimensions": 1024,
                    "normalize": True
                })
            )
            
            response_body = json.loads(test_response['body'].read())
            embedding_dim = len(response_body.get('embedding', []))
            
            results["bedrock"] = {
                "status": "healthy",
                "embedding_dimensions": embedding_dim,
                "message": f"✅ Bedrock connected (Titan v2: {embedding_dim}D embeddings)"
            }
            logger.info(results["bedrock"]["message"])
            
        except Exception as e:
            results["bedrock"] = {
                "status": "unhealthy",
                "error": str(e),
                "message": f"❌ Bedrock connection failed: {e}"
            }
            logger.error(results["bedrock"]["message"])
        
        return results
    
    def generate_embedding(self, text: str, normalize: bool = True) -> List[float]:
        """
        Generate 1024-dimensional embedding using Titan v2.
        
        Args:
            text: Input text
            normalize: Whether to normalize the vector
            
        Returns:
            1024-dimensional embedding vector
        """
        try:
            body = json.dumps({
                "inputText": text,
                "dimensions": 1024,
                "normalize": normalize
            })
            
            response = self.bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding')
            
            if not embedding or len(embedding) != 1024:
                raise AWSClientError(f"Invalid embedding dimension: {len(embedding) if embedding else 0}")
            
            return embedding
            
        except ClientError as e:
            logger.error(f"Bedrock embedding generation failed: {e}")
            raise AWSClientError(f"Failed to generate embedding: {e}") from e
    
    def invoke_claude(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """
        Invoke Claude 3.5 Sonnet for text generation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated text response
        """
        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            
            if system_prompt:
                body["system"] = system_prompt
            
            response = self.bedrock_runtime.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            content = response_body.get('content', [])
            
            if content and len(content) > 0:
                return content[0].get('text', '')
            
            raise AWSClientError("Empty response from Claude")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                logger.warning("Bedrock throttling detected")
                raise AWSClientError("Service is busy, please try again") from e
            else:
                logger.error(f"Claude invocation failed: {e}")
                raise AWSClientError(f"Failed to invoke Claude: {e}") from e
    
    def create_knn_index(self, index_name: Optional[str] = None) -> bool:
        """
        Create OpenSearch index with k-NN support for 1024D vectors.
        
        Args:
            index_name: Index name (defaults to self.opensearch_index)
            
        Returns:
            True if created, False if already exists
        """
        if not self.opensearch_client:
            raise AWSClientError("OpenSearch client not initialized")
        
        index_name = index_name or self.opensearch_index
        
        # Check if index exists
        if self.opensearch_client.indices.exists(index=index_name):
            logger.info(f"Index '{index_name}' already exists")
            return False
        
        # Index configuration for Titan v2 (1024 dimensions)
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "number_of_shards": 2,
                    "number_of_replicas": 1
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1024,
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
                    "category": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "source_url": {"type": "keyword"},
                    "author": {"type": "text"},
                    "stars": {"type": "integer"},
                    "downloads": {"type": "integer"},
                    "license": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "last_updated": {"type": "date"}
                }
            }
        }
        
        try:
            response = self.opensearch_client.indices.create(
                index=index_name,
                body=index_body
            )
            logger.info(f"✅ Index '{index_name}' created with 1024D k-NN support")
            return response.get('acknowledged', False)
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise AWSClientError(f"Index creation failed: {e}") from e
    
    def index_document(
        self,
        document: Dict[str, Any],
        doc_id: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Index a document with embedding in OpenSearch.
        
        Args:
            document: Document to index (must include 'embedding' field)
            doc_id: Optional document ID
            index_name: Index name (defaults to self.opensearch_index)
            
        Returns:
            Indexing response
        """
        if not self.opensearch_client:
            raise AWSClientError("OpenSearch client not initialized")
        
        index_name = index_name or self.opensearch_index
        
        try:
            response = self.opensearch_client.index(
                index=index_name,
                body=document,
                id=doc_id,
                refresh=False  # Bulk operations should not refresh
            )
            return response
            
        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            raise AWSClientError(f"Document indexing failed: {e}") from e
    
    def knn_search(
        self,
        query_vector: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform k-NN vector search.
        
        Args:
            query_vector: Query embedding (1024D)
            k: Number of nearest neighbors
            filters: Optional filters
            index_name: Index name
            
        Returns:
            List of matching documents with scores
        """
        if not self.opensearch_client:
            raise AWSClientError("OpenSearch client not initialized")
        
        index_name = index_name or self.opensearch_index
        
        query_body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            }
        }
        
        # Add filters if provided
        if filters:
            query_body["query"] = {
                "bool": {
                    "must": [{"knn": query_body["query"]["knn"]}],
                    "filter": [{"term": {field: value}} for field, value in filters.items()]
                }
            }
        
        try:
            response = self.opensearch_client.search(
                index=index_name,
                body=query_body
            )
            
            hits = response.get('hits', {}).get('hits', [])
            results = [
                {
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'document': hit['_source']
                }
                for hit in hits
            ]
            
            return results
            
        except Exception as e:
            logger.error(f"k-NN search failed: {e}")
            raise AWSClientError(f"Search failed: {e}") from e
