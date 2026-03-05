"""
Embedding Service

Generates embeddings using Amazon Bedrock
"""
import hashlib
import logging
from typing import List, Optional
import boto3
import json
import redis
from config import settings, ingestion_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings using Amazon Bedrock.
    
    Features:
    - Caching in Redis to avoid redundant API calls
    - Batch processing support
    - Error handling and retries
    """
    
    def __init__(self):
        self.bedrock_client = None
        self.redis_client = None
        self.model_id = ingestion_settings.embedding_model_id
        self.dimensions = ingestion_settings.embedding_dimensions
        self.cache_ttl = ingestion_settings.embedding_cache_ttl
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Bedrock and Redis clients"""
        try:
            # Initialize Bedrock client
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
            logger.info("Initialized Bedrock client")
            
            # Initialize Redis client for caching
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=True,
                socket_timeout=5
            )
            self.redis_client.ping()
            logger.info("Initialized Redis client for embedding cache")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        # Check cache first
        text_hash = self._hash_text(text)
        cached_embedding = self._get_cached_embedding(text_hash)
        if cached_embedding:
            logger.debug("Using cached embedding")
            return cached_embedding
        
        # Generate embedding via Bedrock
        try:
            embedding = await self._generate_bedrock_embedding(text)
            
            if embedding:
                # Cache the embedding
                self._cache_embedding(text_hash, embedding)
                return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
        
        return None
    
    async def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        embeddings = []
        
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    async def _generate_bedrock_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding using Amazon Bedrock.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Prepare request body based on model
            if 'titan' in self.model_id.lower():
                # Amazon Titan Embeddings
                body = json.dumps({
                    "inputText": text[:8000]  # Titan limit
                })
            elif 'cohere' in self.model_id.lower():
                # Cohere Embed
                body = json.dumps({
                    "texts": [text[:2048]],  # Cohere limit
                    "input_type": "search_document"
                })
            else:
                logger.error(f"Unsupported embedding model: {self.model_id}")
                return None
            
            # Invoke Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            if 'titan' in self.model_id.lower():
                embedding = response_body.get('embedding')
            elif 'cohere' in self.model_id.lower():
                embeddings = response_body.get('embeddings', [])
                embedding = embeddings[0] if embeddings else None
            else:
                embedding = None
            
            if embedding and len(embedding) == self.dimensions:
                return embedding
            else:
                logger.warning(f"Invalid embedding dimensions: expected {self.dimensions}, got {len(embedding) if embedding else 0}")
                return None
            
        except Exception as e:
            logger.error(f"Bedrock API error: {e}")
            return None
    
    def _hash_text(self, text: str) -> str:
        """
        Generate hash for text.
        
        Args:
            text: Input text
            
        Returns:
            SHA256 hash
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """
        Get cached embedding from Redis.
        
        Args:
            text_hash: Text hash
            
        Returns:
            Cached embedding or None
        """
        try:
            key = f"embedding:{text_hash}"
            cached = self.redis_client.get(key)
            
            if cached:
                data = json.loads(cached)
                return data.get('embedding')
            
        except Exception as e:
            logger.error(f"Error getting cached embedding: {e}")
        
        return None
    
    def _cache_embedding(self, text_hash: str, embedding: List[float]):
        """
        Cache embedding in Redis.
        
        Args:
            text_hash: Text hash
            embedding: Embedding vector
        """
        try:
            key = f"embedding:{text_hash}"
            value = json.dumps({'embedding': embedding})
            self.redis_client.setex(key, self.cache_ttl, value)
            logger.debug(f"Cached embedding for hash {text_hash[:8]}...")
        except Exception as e:
            logger.error(f"Error caching embedding: {e}")
    
    async def close(self):
        """Close connections"""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Closed Redis connection")
