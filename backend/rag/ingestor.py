"""
Production-ready data ingestion pipeline for Dev-Store RAG system.

This module handles:
- Reading JSON datasets from local files
- Validating data with Pydantic models
- Chunking large descriptions
- Generating embeddings via Bedrock Titan
- Upserting to OpenSearch with proper metadata
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ResourceSchema(BaseModel):
    """Pydantic model for validating resource data before ingestion"""
    name: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=1)
    source: str = Field(..., pattern="^(github|huggingface|kaggle)$")
    source_url: str
    author: str
    stars: int = Field(default=0, ge=0)
    downloads: int = Field(default=0, ge=0)
    license: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    category: str = Field(..., pattern="^(api|model|dataset)$")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('description')
    def description_not_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError('Description cannot be empty')
        return v.strip()
    
    @validator('tags', pre=True)
    def parse_tags(cls, v):
        if isinstance(v, str):
            return [tag.strip() for tag in v.split(',') if tag.strip()]
        return v or []


class DataIngestor:
    """
    Handles automated ingestion of resources from JSON files.
    
    Features:
    - Schema validation with Pydantic
    - Text chunking for large descriptions
    - Batch embedding generation
    - Upsert to OpenSearch with retry logic
    """
    
    def __init__(
        self,
        bedrock_client,
        opensearch_client,
        chunk_size: int = 500,
        batch_size: int = 10
    ):
        """
        Initialize the data ingestor.
        
        Args:
            bedrock_client: BedrockClient instance for embeddings
            opensearch_client: OpenSearchClient instance
            chunk_size: Maximum characters per chunk
            batch_size: Number of documents to process in batch
        """
        self.bedrock = bedrock_client
        self.opensearch = opensearch_client
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        
    def load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load and parse JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} records from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    def validate_resource(self, resource: Dict[str, Any]) -> Optional[ResourceSchema]:
        """Validate resource against schema"""
        try:
            return ResourceSchema(**resource)
        except Exception as e:
            logger.warning(f"Validation failed for {resource.get('name', 'unknown')}: {e}")
            return None
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for better embedding quality.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split by sentences (simple approach)
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:self.chunk_size]]
    
    def create_searchable_text(self, resource: ResourceSchema) -> str:
        """Create combined text for embedding generation"""
        parts = [
            f"Name: {resource.name}",
            f"Description: {resource.description}",
            f"Category: {resource.category}",
            f"Source: {resource.source}",
        ]
        
        if resource.tags:
            parts.append(f"Tags: {', '.join(resource.tags[:10])}")
        
        return " | ".join(parts)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Bedrock Titan v2"""
        try:
            return self.bedrock.generate_embedding(text)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def prepare_document(
        self,
        resource: ResourceSchema,
        embedding: List[float]
    ) -> Dict[str, Any]:
        """
        Prepare document for OpenSearch indexing.
        
        Args:
            resource: Validated resource
            embedding: Generated embedding vector
            
        Returns:
            Document ready for indexing
        """
        return {
            'name': resource.name,
            'description': resource.description[:1000],  # Limit description length
            'resource_type': resource.category.upper(),
            'pricing_type': 'free' if resource.license and 'mit' in resource.license.lower() else 'paid',
            'source': resource.source,
            'source_url': resource.source_url,
            'author': resource.author,
            'github_stars': resource.stars,
            'downloads': resource.downloads,
            'license': resource.license or 'Unknown',
            'tags': resource.tags[:20],  # Limit tags
            'last_updated': datetime.utcnow().isoformat(),
            'health_status': 'healthy',
            'embedding': embedding,
            'metadata': resource.metadata
        }
    
    def ingest_file(
        self,
        file_path: Path,
        skip_existing: bool = True
    ) -> Dict[str, int]:
        """
        Ingest a single JSON file.
        
        Args:
            file_path: Path to JSON file
            skip_existing: Skip if index already has documents
            
        Returns:
            Statistics dict with counts
        """
        stats = {
            'total': 0,
            'validated': 0,
            'indexed': 0,
            'failed': 0
        }
        
        # Load data
        raw_data = self.load_json_file(file_path)
        stats['total'] = len(raw_data)
        
        # Validate and process in batches
        batch = []
        
        for item in raw_data:
            # Validate
            resource = self.validate_resource(item)
            if not resource:
                stats['failed'] += 1
                continue
            
            stats['validated'] += 1
            batch.append(resource)
            
            # Process batch
            if len(batch) >= self.batch_size:
                indexed = self._process_batch(batch)
                stats['indexed'] += indexed
                stats['failed'] += len(batch) - indexed
                batch = []
        
        # Process remaining
        if batch:
            indexed = self._process_batch(batch)
            stats['indexed'] += indexed
            stats['failed'] += len(batch) - indexed
        
        logger.info(f"Ingestion complete for {file_path.name}: {stats}")
        return stats
    
    def _process_batch(self, resources: List[ResourceSchema]) -> int:
        """Process a batch of resources"""
        indexed_count = 0
        
        for resource in resources:
            try:
                # Create searchable text
                text = self.create_searchable_text(resource)
                
                # Generate embedding
                embedding = self.generate_embedding(text)
                
                # Prepare document
                document = self.prepare_document(resource, embedding)
                
                # Index in OpenSearch
                self.opensearch.index_document(
                    document=document,
                    refresh=False  # Batch refresh later
                )
                
                indexed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {resource.name}: {e}")
                continue
        
        return indexed_count
    
    def ingest_all(
        self,
        data_dir: Path = Path("backend")
    ) -> Dict[str, Any]:
        """
        Ingest all JSON files from data directory.
        
        Args:
            data_dir: Directory containing JSON files
            
        Returns:
            Combined statistics
        """
        json_files = [
            data_dir / "github_resources.json",
            data_dir / "huggingface_datasets.json",
            data_dir / "kaggle_datasets.json",
            data_dir / "models.json"
        ]
        
        total_stats = {
            'files_processed': 0,
            'total_records': 0,
            'total_indexed': 0,
            'total_failed': 0
        }
        
        for file_path in json_files:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                stats = self.ingest_file(file_path)
                total_stats['files_processed'] += 1
                total_stats['total_records'] += stats['total']
                total_stats['total_indexed'] += stats['indexed']
                total_stats['total_failed'] += stats['failed']
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                continue
        
        logger.info(f"All files ingested: {total_stats}")
        return total_stats
