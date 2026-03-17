"""
Embedding Service

Generates embeddings using Amazon Bedrock
"""
import hashlib
from typing import List, Dict, Any, Optional
from cachetools import TTLCache


class EmbeddingService:
    """
    Service for generating embeddings via Bedrock
    
    Features:
    - Content-based caching in memory
    - Batch processing
    - Change detection
    """
    
    def __init__(self, bedrock_client):
        """
        Initialize embedding service
        
        Args:
            bedrock_client: Bedrock client (from clients/bedrock.py)
        """
        self.bedrock = bedrock_client
        self.cache = TTLCache(maxsize=5000, ttl=86400)
        self.model_id = "amazon.titan-embed-text-v1"
        
        from ingestion.services.chunking_service import ChunkingService
        self.chunker = ChunkingService()
    
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
        if use_cache:
            cached = self._get_cached_embedding(text)
            if cached:
                return cached
        
        # Generate embedding via Bedrock
        try:
            embedding = self.bedrock.generate_embedding(text)
            
            # Cache result
            if use_cache and embedding:
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

    def generate_chunked_batch(
        self,
        resources: list[dict],
        batch_size: int = 25
    ) -> list[dict]:
        """
        Generate embeddings for resources by breaking them into RAG chunks first.
        
        Args:
            resources: Master list of documents
            batch_size: Batch size for embedding API throttling
            
        Returns:
            List of chunk records padded with 'embedding' vectors. 
            Suitable for Pinecone ingestion.
        """
        # 1. Chunk all resources
        chunked_resources = []
        for resource in resources:
            chunks = self.chunker.split_resource(resource)
            chunked_resources.extend(chunks)
            
        # 2. Embed all chunks
        embedded_chunks = []
        for i in range(0, len(chunked_resources), batch_size):
            batch = chunked_resources[i:i + batch_size]
            for chunk in batch:
                embedding = self.generate_embedding(chunk)
                if embedding:
                    chunk["embedding"] = embedding
                    embedded_chunks.append(chunk)
                    
        return embedded_chunks
    
    def _generate_embedding_text(self, resource: Dict[str, Any]) -> str:
        """
        Generate text for embedding from resource fields.
        Prioritizes RAG-chunked 'text_content' if available.
        """
        if resource.get('text_content'):
            return resource['text_content']
            
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
        cache_key = self._get_cache_key(text)
        return self.cache.get(cache_key)
    
    def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding in Memory"""
        cache_key = self._get_cache_key(text)
        self.cache[cache_key] = embedding
