"""
Production Ingestion Orchestrator

Uses existing infrastructure components:
- DatabaseClient for PostgreSQL operations
- BedrockClient for embeddings
- OpenSearchClient for indexing
- RedisClient for caching and locking
- IngestionRepository for database operations
- EmbeddingService for embedding generation
- IndexingService for OpenSearch operations
- RankingService for score computation
"""
import sys
import os
import uuid
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.database import DatabaseClient
from clients.bedrock import BedrockClient
from clients.opensearch import OpenSearchClient
from clients.redis_client import RedisClient
from ingestion.fetchers.huggingface_fetcher import HuggingFaceFetcher
from ingestion.fetchers.openrouter_fetcher import OpenRouterFetcher
from ingestion.fetchers.github_fetcher import GitHubFetcher
from ingestion.fetchers.kaggle_fetcher import KaggleFetcher
from ingestion.repository import IngestionRepository
from models.ingestion import CanonicalResource, IngestionSource, IngestionStatus, ResourceType, PricingType, HealthStatus


class ProductionOrchestrator:
    """
    Production ingestion orchestrator using real infrastructure
    
    Pipeline stages:
    1. Acquire Lock - Prevent concurrent runs
    2. Fetch - Get data from external sources
    3. Normalize - Convert to canonical schema
    4. Deduplicate - Remove duplicates
    5. Upsert - Insert/update in PostgreSQL
    6. Embed - Generate embeddings via Bedrock
    7. Index - Update OpenSearch
    8. Rank - Recompute rankings
    9. Invalidate - Clear Redis caches
    10. Release Lock - Allow next run
    """
    
    def __init__(
        self,
        run_id: Optional[str] = None,
        sources: Optional[List[str]] = None,
        log_level: str = "INFO",
        use_infrastructure: bool = True
    ):
        """
        Initialize orchestrator
        
        Args:
            run_id: Unique run identifier (generated if not provided)
            sources: List of sources to run (all if None)
            log_level: Logging level
            use_infrastructure: Whether to use real infrastructure (DB, Redis, etc.)
        """
        self.run_id = run_id or str(uuid.uuid4())
        self.sources = sources or ['huggingface', 'openrouter', 'github', 'kaggle']
        self.started_at = datetime.utcnow()
        self.finished_at = None
        self.use_infrastructure = use_infrastructure
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Initialize clients (only if using infrastructure)
        self.db_client = None
        self.bedrock_client = None
        self.opensearch_client = None
        self.redis_client = None
        self.repository = None
        self.lock_token = None
        
        if use_infrastructure:
            self._initialize_clients()
        
        # Statistics
        self.stats = {
            'fetched_count': 0,
            'inserted_count': 0,
            'updated_count': 0,
            'failed_count': 0,
            'skipped_count': 0,
            'embedded_count': 0,
            'indexed_count': 0,
        }
        
        # Per-source stats
        self.source_stats = {}
        
        self.logger.info(f"Ingestion run started: {self.run_id}")
        self.logger.info(f"Sources enabled: {', '.join(self.sources)}")
        self.logger.info(f"Infrastructure mode: {'ENABLED' if use_infrastructure else 'DISABLED (JSON only)'}")
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup structured logging with run ID"""
        logger = logging.getLogger(f"ingestion.{self.run_id}")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        log_dir = Path(__file__).parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"ingestion_{self.run_id}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter with run ID
        formatter = logging.Formatter(
            f'%(asctime)s - {self.run_id} - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_clients(self):
        """Initialize infrastructure clients"""
        try:
            self.logger.info("Initializing infrastructure clients...")
            
            # Database
            self.db_client = DatabaseClient()
            self.repository = IngestionRepository(self.db_client)
            self.logger.info("✓ Database client initialized")
            
            # Bedrock
            self.bedrock_client = BedrockClient()
            self.logger.info("✓ Bedrock client initialized")
            
            # OpenSearch
            self.opensearch_client = OpenSearchClient()
            self.logger.info("✓ OpenSearch client initialized")
            
            # Redis
            self.redis_client = RedisClient()
            # Note: Redis client needs async connect, will be called in async context
            self.logger.info("✓ Redis client initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize clients: {e}")
            self.logger.warning("Falling back to JSON-only mode")
            self.use_infrastructure = False
    
    async def run(self) -> Dict[str, Any]:
        """
        Execute complete ingestion pipeline
        
        Returns:
            Run statistics and status
        """
        try:
            self.logger.info("=" * 70)
            self.logger.info("PRODUCTION INGESTION PIPELINE STARTED")
            self.logger.info("=" * 70)
            
            # Stage 0: Acquire distributed lock (if using Redis)
            if self.use_infrastructure and self.redis_client:
                if not await self._acquire_lock():
                    return {
                        'run_id': self.run_id,
                        'status': 'skipped',
                        'reason': 'Another ingestion run is in progress',
                        'started_at': self.started_at.isoformat(),
                    }
            
            # Stage 1: Fetch & Normalize
            all_resources = await self._stage_fetch_and_normalize()
            
            # Stage 2: Deduplicate
            deduplicated = self._stage_deduplicate(all_resources)
            
            if self.use_infrastructure:
                # Stage 3: Upsert to PostgreSQL
                upserted = await self._stage_upsert(deduplicated)
                
                # Stage 4: Generate Embeddings
                embedded = await self._stage_embed(upserted)
                
                # Stage 5: Index in OpenSearch
                indexed = await self._stage_index(embedded)
                
                # Stage 6: Refresh Rankings
                await self._stage_rank()
                
                # Stage 7: Invalidate Caches
                await self._stage_invalidate_cache()
            else:
                # JSON-only mode: save to files
                self._save_to_json(deduplicated)
            
            self.finished_at = datetime.utcnow()
            duration = (self.finished_at - self.started_at).total_seconds()
            
            self.logger.info("=" * 70)
            self.logger.info("INGESTION PIPELINE COMPLETED")
            self.logger.info(f"Duration: {duration:.2f} seconds")
            self.logger.info(f"Fetched: {self.stats['fetched_count']}")
            self.logger.info(f"Inserted: {self.stats['inserted_count']}")
            self.logger.info(f"Updated: {self.stats['updated_count']}")
            self.logger.info(f"Embedded: {self.stats['embedded_count']}")
            self.logger.info(f"Indexed: {self.stats['indexed_count']}")
            self.logger.info(f"Failed: {self.stats['failed_count']}")
            self.logger.info("=" * 70)
            
            return {
                'run_id': self.run_id,
                'status': 'completed',
                'started_at': self.started_at.isoformat(),
                'finished_at': self.finished_at.isoformat(),
                'duration_seconds': duration,
                'stats': self.stats,
                'source_stats': self.source_stats,
            }
            
        except Exception as e:
            self.logger.error(f"Ingestion pipeline failed: {e}", exc_info=True)
            self.finished_at = datetime.utcnow()
            
            return {
                'run_id': self.run_id,
                'status': 'failed',
                'error': str(e),
                'started_at': self.started_at.isoformat(),
                'finished_at': self.finished_at.isoformat(),
                'stats': self.stats,
            }
        finally:
            # Release lock
            if self.use_infrastructure and self.redis_client and self.lock_token:
                await self._release_lock()
    
    async def _acquire_lock(self) -> bool:
        """Acquire distributed lock to prevent concurrent runs"""
        self.logger.info("Acquiring distributed lock...")
        
        try:
            await self.redis_client.connect()
            lock_key = "ingestion:lock"
            lock_ttl = 3600  # 1 hour
            
            self.lock_token = await self.redis_client.acquire_lock(lock_key, lock_ttl)
            
            if self.lock_token:
                self.logger.info("✓ Lock acquired")
                return True
            else:
                self.logger.warning("✗ Lock already held by another process")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to acquire lock: {e}")
            return False
    
    async def _release_lock(self):
        """Release distributed lock"""
        self.logger.info("Releasing distributed lock...")
        
        try:
            lock_key = "ingestion:lock"
            released = await self.redis_client.release_lock(lock_key, self.lock_token)
            
            if released:
                self.logger.info("✓ Lock released")
            else:
                self.logger.warning("✗ Failed to release lock (may have expired)")
                
        except Exception as e:
            self.logger.error(f"Error releasing lock: {e}")
    
    async def _stage_fetch_and_normalize(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Stage 1: Fetch data from all enabled sources and normalize
        
        Returns:
            Dictionary of resources by type
        """
        self.logger.info("Stage 1: Fetch & Normalize")
        self.logger.info("-" * 70)
        
        all_resources = {
            'models': [],
            'datasets': [],
            'repositories': [],
        }
        
        # Fetch from HuggingFace
        if 'huggingface' in self.sources:
            try:
                self.logger.info("Fetching from HuggingFace...")
                fetcher = HuggingFaceFetcher()
                results = fetcher.fetch_and_normalize_all()
                
                all_resources['models'].extend(results['models'])
                all_resources['datasets'].extend(results['datasets'])
                
                self.source_stats['huggingface'] = {
                    'models': len(results['models']),
                    'datasets': len(results['datasets']),
                }
                self.stats['fetched_count'] += len(results['models']) + len(results['datasets'])
                
                self.logger.info(f"✓ HuggingFace: {len(results['models'])} models, {len(results['datasets'])} datasets")
            except Exception as e:
                self.logger.error(f"✗ HuggingFace fetch failed: {e}")
                self.stats['failed_count'] += 1
        
        # Fetch from OpenRouter
        if 'openrouter' in self.sources:
            try:
                self.logger.info("Fetching from OpenRouter...")
                fetcher = OpenRouterFetcher()
                models = fetcher.fetch_and_normalize_all()
                
                all_resources['models'].extend(models)
                
                self.source_stats['openrouter'] = {'models': len(models)}
                self.stats['fetched_count'] += len(models)
                
                self.logger.info(f"✓ OpenRouter: {len(models)} models")
            except Exception as e:
                self.logger.error(f"✗ OpenRouter fetch failed: {e}")
                self.stats['failed_count'] += 1
        
        # Fetch from GitHub
        if 'github' in self.sources:
            try:
                self.logger.info("Fetching from GitHub...")
                fetcher = GitHubFetcher()
                repos = fetcher.fetch_and_normalize_all()
                
                all_resources['repositories'].extend(repos)
                
                self.source_stats['github'] = {'repositories': len(repos)}
                self.stats['fetched_count'] += len(repos)
                
                self.logger.info(f"✓ GitHub: {len(repos)} repositories")
            except Exception as e:
                self.logger.error(f"✗ GitHub fetch failed: {e}")
                self.stats['failed_count'] += 1
        
        # Fetch from Kaggle
        if 'kaggle' in self.sources:
            try:
                self.logger.info("Fetching from Kaggle...")
                fetcher = KaggleFetcher()
                datasets = fetcher.fetch_and_normalize_all(max_pages=5)
                
                all_resources['datasets'].extend(datasets)
                
                self.source_stats['kaggle'] = {'datasets': len(datasets)}
                self.stats['fetched_count'] += len(datasets)
                
                self.logger.info(f"✓ Kaggle: {len(datasets)} datasets")
            except ImportError:
                self.logger.warning("⚠ Kaggle package not installed, skipping")
            except Exception as e:
                self.logger.error(f"✗ Kaggle fetch failed: {e}")
                self.stats['failed_count'] += 1
        
        self.logger.info(f"Total fetched: {self.stats['fetched_count']} resources")
        return all_resources
    
    def _stage_deduplicate(self, resources: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Stage 2: Deduplicate resources
        
        Deduplication key: source + source_url
        """
        self.logger.info("\nStage 2: Deduplicate")
        self.logger.info("-" * 70)
        
        # Combine all resources
        all_resources = (
            resources['models'] +
            resources['datasets'] +
            resources['repositories']
        )
        
        before_count = len(all_resources)
        
        # Deduplicate by source + source_url
        seen = {}
        deduplicated = []
        
        for resource in all_resources:
            key = f"{resource['source']}:{resource['source_url']}"
            if key not in seen:
                seen[key] = resource
                deduplicated.append(resource)
        
        after_count = len(deduplicated)
        removed = before_count - after_count
        
        self.stats['skipped_count'] = removed
        self.logger.info(f"Resources: {before_count} → {after_count} (removed {removed} duplicates)")
        
        return deduplicated
    
    async def _stage_upsert(self, resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 3: Upsert to PostgreSQL"""
        self.logger.info("\nStage 3: Upsert to PostgreSQL")
        self.logger.info("-" * 70)
        
        upserted_resources = []
        
        for resource in resources:
            try:
                # Convert to CanonicalResource model
                canonical = self._to_canonical_resource(resource)
                
                # Upsert via repository
                result = self.repository.upsert_resource(canonical)
                
                if result.action == 'inserted':
                    self.stats['inserted_count'] += 1
                elif result.action == 'updated':
                    self.stats['updated_count'] += 1
                
                # Add resource ID for next stages
                resource['id'] = result.resource_id
                resource['embedding_changed'] = result.embedding_changed
                upserted_resources.append(resource)
                
            except Exception as e:
                self.logger.error(f"Failed to upsert resource {resource.get('name')}: {e}")
                self.stats['failed_count'] += 1
        
        self.logger.info(f"Inserted: {self.stats['inserted_count']}, Updated: {self.stats['updated_count']}")
        return upserted_resources
    
    async def _stage_embed(self, resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 4: Generate embeddings via Bedrock"""
        self.logger.info("\nStage 4: Generate Embeddings")
        self.logger.info("-" * 70)
        
        # Filter resources that need embeddings
        needs_embedding = [r for r in resources if r.get('embedding_changed', True)]
        
        self.logger.info(f"Resources needing embeddings: {len(needs_embedding)}")
        
        for resource in needs_embedding:
            try:
                # Generate embedding text
                embedding_text = self._generate_embedding_text(resource)
                
                # Check cache first
                text_hash = hashlib.sha256(embedding_text.encode()).hexdigest()
                cached_embedding = await self.redis_client.get_cached_embedding(text_hash)
                
                if cached_embedding:
                    embedding = cached_embedding
                    self.logger.debug(f"Using cached embedding for {resource['name']}")
                else:
                    # Generate via Bedrock
                    embedding = self.bedrock_client.generate_embedding(embedding_text)
                    
                    # Cache it
                    await self.redis_client.cache_embedding(text_hash, embedding, ttl=86400 * 30)
                
                # Update in database
                self.repository.update_embedding(
                    resource['id'],
                    embedding,
                    resource.get('content_hash', '')
                )
                
                resource['embedding'] = embedding
                self.stats['embedded_count'] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to generate embedding for {resource.get('name')}: {e}")
                self.stats['failed_count'] += 1
        
        self.logger.info(f"Embeddings generated: {self.stats['embedded_count']}")
        return resources
    
    async def _stage_index(self, resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 5: Index in OpenSearch"""
        self.logger.info("\nStage 5: Index in OpenSearch")
        self.logger.info("-" * 70)
        
        # Filter resources with embeddings
        to_index = [r for r in resources if r.get('embedding')]
        
        self.logger.info(f"Resources to index: {len(to_index)}")
        
        # Batch index
        batch_size = 100
        for i in range(0, len(to_index), batch_size):
            batch = to_index[i:i + batch_size]
            
            try:
                # Prepare documents
                documents = []
                for resource in batch:
                    doc = self._to_opensearch_document(resource)
                    documents.append(doc)
                
                # Bulk index
                # Note: This is a simplified version, real implementation would use opensearch-py bulk API
                for doc in documents:
                    self.opensearch_client.index_document(
                        document=doc,
                        doc_id=doc['id']
                    )
                    self.stats['indexed_count'] += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to index batch: {e}")
                self.stats['failed_count'] += len(batch)
        
        self.logger.info(f"Resources indexed: {self.stats['indexed_count']}")
        return resources
    
    async def _stage_rank(self):
        """Stage 6: Refresh rankings"""
        self.logger.info("\nStage 6: Refresh Rankings")
        self.logger.info("-" * 70)
        
        try:
            # Fetch all resources for ranking
            resources = self.repository.fetch_resources_for_ranking()
            
            # Compute rankings
            rankings = []
            for resource in resources:
                rank_score = self._compute_rank_score(resource)
                trending_score = self._compute_trending_score(resource)
                
                rankings.append({
                    'id': resource['id'],
                    'rank_score': rank_score,
                    'trending_score': trending_score,
                    'category_rank': 0,  # Will be computed in persist_rankings
                    'popularity_score': rank_score,
                    'optimization_score': 0.5,
                    'freshness_score': trending_score / rank_score if rank_score > 0 else 0,
                    'final_score': rank_score,
                    'rank_position': 0,
                })
            
            # Sort by rank_score
            rankings.sort(key=lambda x: x['rank_score'], reverse=True)
            
            # Assign positions
            for i, ranking in enumerate(rankings):
                ranking['rank_position'] = i + 1
            
            # Persist rankings
            result = self.repository.persist_rankings(rankings)
            
            self.logger.info(f"✓ Rankings updated: {result['ranked_count']} resources")
            
        except Exception as e:
            self.logger.error(f"Failed to refresh rankings: {e}")
    
    async def _stage_invalidate_cache(self):
        """Stage 7: Invalidate Redis caches"""
        self.logger.info("\nStage 7: Invalidate Caches")
        self.logger.info("-" * 70)
        
        try:
            # Invalidate search caches
            await self.redis_client.invalidate_pattern("search:*")
            
            # Invalidate ranking caches
            await self.redis_client.invalidate_pattern("ranking:*")
            
            # Invalidate resource caches
            await self.redis_client.invalidate_pattern("resource:*")
            
            self.logger.info("✓ Caches invalidated")
            
        except Exception as e:
            self.logger.error(f"Failed to invalidate caches: {e}")
    
    def _save_to_json(self, resources: List[Dict[str, Any]]):
        """Save resources to JSON files (fallback mode)"""
        self.logger.info("\nSaving to JSON files...")
        
        import json
        from pathlib import Path
        
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(exist_ok=True)
        
        # Separate by type
        models = [r for r in resources if r.get('category') == 'model']
        datasets = [r for r in resources if r.get('category') == 'dataset']
        repos = [r for r in resources if r.get('category') == 'solution']
        
        # Save models
        with open(output_dir / 'models.json', 'w', encoding='utf-8') as f:
            json.dump(models, f, indent=2, default=str)
        
        # Save datasets
        with open(output_dir / 'datasets.json', 'w', encoding='utf-8') as f:
            json.dump(datasets, f, indent=2, default=str)
        
        # Save repositories
        with open(output_dir / 'repositories.json', 'w', encoding='utf-8') as f:
            json.dump(repos, f, indent=2, default=str)
        
        self.logger.info(f"✓ Saved {len(models)} models, {len(datasets)} datasets, {len(repos)} repositories")
    
    def _to_canonical_resource(self, resource: Dict[str, Any]) -> CanonicalResource:
        """Convert fetcher output to CanonicalResource model"""
        # Map category to resource_type
        category = resource.get('category', 'solution')
        if category == 'model':
            resource_type = ResourceType.MODEL
        elif category == 'api':
            resource_type = ResourceType.API
        elif category == 'dataset':
            resource_type = ResourceType.DATASET
        else:
            resource_type = ResourceType.SOLUTION
        
        return CanonicalResource(
            source=IngestionSource(resource['source']),
            resource_type=resource_type,
            name=resource['name'],
            description=resource.get('description', ''),
            source_url=resource['source_url'],
            documentation_url=resource.get('readme_url'),
            pricing_type=PricingType.FREE,  # Default, can be enhanced
            github_stars=resource.get('stars', 0),
            download_count=resource.get('downloads', 0),
            active_users=0,
            health_status=HealthStatus.HEALTHY,
            tags=resource.get('tags', []),
            categories=[category],
            metadata=resource.get('metadata', {}),
            source_updated_at=None,
            raw_payload=resource
        )
    
    def _generate_embedding_text(self, resource: Dict[str, Any]) -> str:
        """Generate text for embedding"""
        parts = []
        
        if resource.get('name'):
            parts.append(f"Name: {resource['name']}")
        
        if resource.get('description'):
            parts.append(f"Description: {resource['description']}")
        
        if resource.get('category'):
            parts.append(f"Category: {resource['category']}")
        
        if resource.get('tags'):
            tags_str = ', '.join(resource['tags'][:10])
            parts.append(f"Tags: {tags_str}")
        
        return ' | '.join(parts)
    
    def _to_opensearch_document(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Convert resource to OpenSearch document"""
        return {
            'id': resource['id'],
            'name': resource['name'],
            'description': resource.get('description', ''),
            'source': resource['source'],
            'source_url': resource['source_url'],
            'author': resource.get('author', ''),
            'stars': resource.get('stars', 0),
            'downloads': resource.get('downloads', 0),
            'license': resource.get('license', 'Unknown'),
            'tags': resource.get('tags', []),
            'category': resource.get('category', 'solution'),
            'embedding': resource.get('embedding', []),
            'rank_score': resource.get('rank_score', 0),
            'trending_score': resource.get('trending_score', 0),
        }
    
    def _compute_rank_score(self, resource: Dict[str, Any]) -> float:
        """Compute overall rank score"""
        import math
        stars = resource.get('github_stars', 0)
        downloads = resource.get('download_count', 0)
        score = math.log(stars + 1) + math.log(downloads + 1) * 0.5
        return round(score, 2)
    
    def _compute_trending_score(self, resource: Dict[str, Any]) -> float:
        """Compute trending score based on recency"""
        rank_score = self._compute_rank_score(resource)
        
        updated_at = resource.get('updated_at')
        if not updated_at:
            return rank_score * 0.1
        
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
        
        days_old = (datetime.utcnow() - updated_at).days
        recency_factor = max(0.1, 1.0 - (days_old / 365.0))
        
        return round(rank_score * recency_factor, 2)


async def main():
    """Main entrypoint for orchestrator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run production ingestion pipeline')
    parser.add_argument('--sources', nargs='+', help='Sources to run')
    parser.add_argument('--log-level', default='INFO', help='Log level')
    parser.add_argument('--run-id', help='Run ID (generated if not provided)')
    parser.add_argument('--no-infrastructure', action='store_true', help='Disable infrastructure (JSON only)')
    
    args = parser.parse_args()
    
    orchestrator = ProductionOrchestrator(
        run_id=args.run_id,
        sources=args.sources,
        log_level=args.log_level,
        use_infrastructure=not args.no_infrastructure
    )
    
    result = await orchestrator.run()
    
    # Exit with error code if failed
    if result['status'] == 'failed':
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
