"""
Embedding Service

Generates embeddings using Amazon Bedrock
"""
import hashlib
from typing import List, Dict, Any, Optional


class EmbeddingService:
    """
    Service for generating embeddings via Bedrock
    
    Features:
    - Content-based caching using Redis
    - Batch processing
    - Change detection
    """
    
    def __init__(self, bedrock_client, redis_client=None):
        """
        Initialize embedding service
        
        Args:
            bedrock_client: Bedrock client (from clients/bedrock.py)
            redis_client: Optional Redis client for caching
        """
        self.bedrock = bedrock_client
        self.redis = redis_client
        self.model_id = "amazon.titan-embed-text-v1"
        self.cache_ttl = 86400 * 30  # 30 days
    
    def generate_embedding(
        self,
        resource: Dict[str, Any],
        use_cache: bool = True
    ) -> Optional[List[float]]:
        """
        Generate embedding for a resource
        
        Args:
            resource: Resource data
            use_cache: Whether to use Redis cache
            
        Returns:
            Embedding vector or None if failed
        """
        # Generate embedding text
        text = self._generate_embedding_text(resource)
        
        # Check cache
        if use_cache and self.redis:
            cached = self._get_cached_embedding(text)
            if cached:
                return cached
        
        # Generate embedding via Bedrock
        try:
            embedding = self.bedrock.generate_embedding(text, self.model_id)
            
            # Cache result
            if use_cache and self.redis and embedding:
                self._cache_embedding(text, embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Failed to generate embedding: {e}")
            return None
    
    def generate_batch(
        self,
        resources: List[Dict[str, Any]],
        batch_size: int = 25
    ) -> Dict[str, List[float]]:
        """
        Generate embeddings for multiple resources
        
        Args:
            resources: List of resources
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping resource_id to embedding vector
        """
        embeddings = {}
        
        for i in range(0, len(resources), batch_size):
            batch = resources[i:i + batch_size]
            
            for resource in batch:
                resource_id = resource['id']
                embedding = self.generate_embedding(resource)
                
                if embedding:
                    embeddings[resource_id] = embedding
        
        return embeddings
    
    def _generate_embedding_text(self, resource: Dict[str, Any]) -> str:
        """
        Generate text for embedding from resource fields
        
        Combines: name, description, tags, category
        """
        parts = []
        
        # Add name
        if resource.get('name'):
            parts.append(f"Name: {resource['name']}")
        
        # Add description
        if resource.get('description'):
            parts.append(f"Description: {resource['description']}")
        
        # Add category
        if resource.get('category'):
            parts.append(f"Category: {resource['category']}")
        
        # Add tags
        if resource.get('tags'):
            tags_str = ', '.join(resource['tags'][:10])
            parts.append(f"Tags: {tags_str}")
        
        return ' | '.join(parts)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"embedding:{text_hash}"
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        if not self.redis:
            return None
        
        cache_key = self._get_cache_key(text)
        cached = self.redis.get(cache_key)
        
        if cached:
            # Parse cached embedding
            import json
            return json.loads(cached)
        
        return None
    
    def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding in Redis"""
        if not self.redis:
            return
        
        cache_key = self._get_cache_key(text)
        
        import json
        self.redis.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(embedding)
        )
