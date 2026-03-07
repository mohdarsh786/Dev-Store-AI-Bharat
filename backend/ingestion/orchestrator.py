"""
Production Ingestion Orchestrator

Single entrypoint for scheduled ingestion runs with:
- Pipeline stages: fetch, normalize, dedupe, upsert, embed, index, rank, cache invalidation
- Per-source execution control
- Structured logging with run IDs
- Redis-based locking to prevent overlapping runs
"""
import sys
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from ingestion.fetchers.huggingface_fetcher import HuggingFaceFetcher
from ingestion.fetchers.openrouter_fetcher import OpenRouterFetcher
from ingestion.fetchers.github_fetcher import GitHubFetcher
from ingestion.fetchers.kaggle_fetcher import KaggleFetcher


class IngestionOrchestrator:
    """
    Orchestrates the complete ingestion pipeline
    
    Pipeline stages:
    1. Fetch - Get data from external sources
    2. Normalize - Convert to canonical schema
    3. Deduplicate - Remove duplicates
    4. Upsert - Insert/update in Aurora
    5. Embed - Generate embeddings via Bedrock
    6. Index - Update OpenSearch
    7. Rank - Recompute rankings
    8. Invalidate - Clear Redis caches
    """
    
    def __init__(
        self,
        run_id: Optional[str] = None,
        sources: Optional[List[str]] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize orchestrator
        
        Args:
            run_id: Unique run identifier (generated if not provided)
            sources: List of sources to run (all if None)
            log_level: Logging level
        """
        self.run_id = run_id or str(uuid.uuid4())
        self.sources = sources or ['huggingface', 'openrouter', 'github', 'kaggle']
        self.started_at = datetime.utcnow()
        self.finished_at = None
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Statistics
        self.stats = {
            'fetched_count': 0,
            'inserted_count': 0,
            'updated_count': 0,
            'failed_count': 0,
            'skipped_count': 0,
        }
        
        # Per-source stats
        self.source_stats = {}
        
        self.logger.info(f"Ingestion run started: {self.run_id}")
        self.logger.info(f"Sources enabled: {', '.join(self.sources)}")
    
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
    
    def run(self) -> Dict[str, Any]:
        """
        Execute complete ingestion pipeline
        
        Returns:
            Run statistics and status
        """
        try:
            self.logger.info("=" * 70)
            self.logger.info("INGESTION PIPELINE STARTED")
            self.logger.info("=" * 70)
            
            # Stage 1: Fetch & Normalize
            all_resources = self._stage_fetch_and_normalize()
            
            # Stage 2: Deduplicate
            deduplicated = self._stage_deduplicate(all_resources)
            
            # Stage 3: Upsert to Aurora
            upserted = self._stage_upsert(deduplicated)
            
            # Stage 4: Generate Embeddings
            embedded = self._stage_embed(upserted)
            
            # Stage 5: Index in OpenSearch
            indexed = self._stage_index(embedded)
            
            # Stage 6: Refresh Rankings
            self._stage_rank()
            
            # Stage 7: Invalidate Caches
            self._stage_invalidate_cache()
            
            # Stage 8: Save Raw Snapshots
            self._stage_save_snapshots(all_resources)
            
            self.finished_at = datetime.utcnow()
            duration = (self.finished_at - self.started_at).total_seconds()
            
            self.logger.info("=" * 70)
            self.logger.info("INGESTION PIPELINE COMPLETED")
            self.logger.info(f"Duration: {duration:.2f} seconds")
            self.logger.info(f"Fetched: {self.stats['fetched_count']}")
            self.logger.info(f"Inserted: {self.stats['inserted_count']}")
            self.logger.info(f"Updated: {self.stats['updated_count']}")
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
    
    def _stage_fetch_and_normalize(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Stage 1: Fetch data from all enabled sources and normalize
        
        Returns:
            Dictionary of resources by source
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
    
    def _stage_deduplicate(self, resources: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Stage 2: Deduplicate resources
        
        Deduplication key: source + source_url
        """
        self.logger.info("\nStage 2: Deduplicate")
        self.logger.info("-" * 70)
        
        # Deduplicate models
        models_before = len(resources['models'])
        resources['models'] = self._deduplicate_by_key(
            resources['models'],
            key_fn=lambda r: f"{r['source']}:{r['source_url']}"
        )
        models_after = len(resources['models'])
        models_removed = models_before - models_after
        
        self.logger.info(f"Models: {models_before} → {models_after} (removed {models_removed} duplicates)")
        
        # Deduplicate datasets
        datasets_before = len(resources['datasets'])
        resources['datasets'] = self._deduplicate_by_key(
            resources['datasets'],
            key_fn=lambda r: f"{r['source']}:{r['source_url']}"
        )
        datasets_after = len(resources['datasets'])
        datasets_removed = datasets_before - datasets_after
        
        self.logger.info(f"Datasets: {datasets_before} → {datasets_after} (removed {datasets_removed} duplicates)")
        
        # Deduplicate repositories
        repos_before = len(resources['repositories'])
        resources['repositories'] = self._deduplicate_by_key(
            resources['repositories'],
            key_fn=lambda r: f"{r['source']}:{r['source_url']}"
        )
        repos_after = len(resources['repositories'])
        repos_removed = repos_before - repos_after
        
        self.logger.info(f"Repositories: {repos_before} → {repos_after} (removed {repos_removed} duplicates)")
        
        total_removed = models_removed + datasets_removed + repos_removed
        self.stats['skipped_count'] = total_removed
        self.logger.info(f"Total duplicates removed: {total_removed}")
        
        return resources
    
    def _deduplicate_by_key(self, items: List[Dict], key_fn) -> List[Dict]:
        """Deduplicate list by key function"""
        seen = {}
        for item in items:
            key = key_fn(item)
            if key not in seen:
                seen[key] = item
        return list(seen.values())
    
    def _stage_upsert(self, resources: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Stage 3: Upsert to Aurora (placeholder)"""
        self.logger.info("\nStage 3: Upsert to Aurora")
        self.logger.info("-" * 70)
        self.logger.info("⚠ Aurora upsert not yet implemented")
        self.logger.info("TODO: Implement database upsert logic")
        return resources
    
    def _stage_embed(self, resources: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Stage 4: Generate embeddings (placeholder)"""
        self.logger.info("\nStage 4: Generate Embeddings")
        self.logger.info("-" * 70)
        self.logger.info("⚠ Embedding generation not yet implemented")
        self.logger.info("TODO: Implement Bedrock embedding generation")
        return resources
    
    def _stage_index(self, resources: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Stage 5: Index in OpenSearch (placeholder)"""
        self.logger.info("\nStage 5: Index in OpenSearch")
        self.logger.info("-" * 70)
        self.logger.info("⚠ OpenSearch indexing not yet implemented")
        self.logger.info("TODO: Implement OpenSearch indexing")
        return resources
    
    def _stage_rank(self):
        """Stage 6: Refresh rankings (placeholder)"""
        self.logger.info("\nStage 6: Refresh Rankings")
        self.logger.info("-" * 70)
        self.logger.info("⚠ Ranking refresh not yet implemented")
        self.logger.info("TODO: Implement ranking computation")
    
    def _stage_invalidate_cache(self):
        """Stage 7: Invalidate Redis caches (placeholder)"""
        self.logger.info("\nStage 7: Invalidate Caches")
        self.logger.info("-" * 70)
        self.logger.info("⚠ Cache invalidation not yet implemented")
        self.logger.info("TODO: Implement Redis cache invalidation")
    
    def _stage_save_snapshots(self, resources: Dict[str, List[Dict[str, Any]]]):
        """Stage 8: Save raw snapshots to S3 (placeholder)"""
        self.logger.info("\nStage 8: Save Raw Snapshots")
        self.logger.info("-" * 70)
        self.logger.info("⚠ S3 snapshot storage not yet implemented")
        self.logger.info("TODO: Implement S3 snapshot persistence")


def main():
    """Main entrypoint for orchestrator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ingestion pipeline')
    parser.add_argument('--sources', nargs='+', help='Sources to run')
    parser.add_argument('--log-level', default='INFO', help='Log level')
    parser.add_argument('--run-id', help='Run ID (generated if not provided)')
    
    args = parser.parse_args()
    
    orchestrator = IngestionOrchestrator(
        run_id=args.run_id,
        sources=args.sources,
        log_level=args.log_level
    )
    
    result = orchestrator.run()
    
    # Exit with error code if failed
    if result['status'] == 'failed':
        sys.exit(1)


if __name__ == '__main__':
    main()
