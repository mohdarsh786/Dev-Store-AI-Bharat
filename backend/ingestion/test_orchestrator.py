"""
Test Orchestrator

Simplified version for testing without full infrastructure
"""
import sys
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fetchers.huggingface_fetcher import HuggingFaceFetcher
from fetchers.openrouter_fetcher import OpenRouterFetcher
from fetchers.github_fetcher import GitHubFetcher
from fetchers.kaggle_fetcher import KaggleFetcher


class TestOrchestrator:
    """
    Simplified orchestrator for testing
    
    Runs fetch and normalize stages only
    Saves results to JSON files
    """
    
    def __init__(self, run_id: Optional[str] = None, sources: Optional[List[str]] = None):
        self.run_id = run_id or str(uuid.uuid4())
        self.sources = sources or ['huggingface', 'openrouter', 'github', 'kaggle']
        self.started_at = datetime.utcnow()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format=f'%(asctime)s - {self.run_id} - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'fetched_count': 0,
            'models_count': 0,
            'datasets_count': 0,
            'repos_count': 0,
        }
        
        self.logger.info(f"Test run started: {self.run_id}")
        self.logger.info(f"Sources: {', '.join(self.sources)}")
    
    def run(self) -> Dict[str, Any]:
        """Execute test pipeline"""
        try:
            self.logger.info("=" * 70)
            self.logger.info("TEST INGESTION PIPELINE")
            self.logger.info("=" * 70)
            
            # Fetch and normalize
            all_resources = self._fetch_all()
            
            # Deduplicate
            deduplicated = self._deduplicate(all_resources)
            
            # Save to files
            self._save_results(deduplicated)
            
            # Summary
            duration = (datetime.utcnow() - self.started_at).total_seconds()
            
            self.logger.info("=" * 70)
            self.logger.info("TEST COMPLETED")
            self.logger.info(f"Duration: {duration:.2f}s")
            self.logger.info(f"Total fetched: {self.stats['fetched_count']}")
            self.logger.info(f"Models: {self.stats['models_count']}")
            self.logger.info(f"Datasets: {self.stats['datasets_count']}")
            self.logger.info(f"Repositories: {self.stats['repos_count']}")
            self.logger.info("=" * 70)
            
            return {
                'status': 'success',
                'stats': self.stats,
                'duration': duration
            }
            
        except Exception as e:
            self.logger.error(f"Test failed: {e}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _fetch_all(self) -> Dict[str, List[Dict]]:
        """Fetch from all sources"""
        self.logger.info("\nFetching from sources...")
        self.logger.info("-" * 70)
        
        results = {
            'models': [],
            'datasets': [],
            'repositories': []
        }
        
        # HuggingFace
        if 'huggingface' in self.sources:
            try:
                self.logger.info("Fetching HuggingFace...")
                fetcher = HuggingFaceFetcher()
                hf_results = fetcher.fetch_and_normalize_all()
                
                results['models'].extend(hf_results['models'])
                results['datasets'].extend(hf_results['datasets'])
                
                self.logger.info(f"✓ HuggingFace: {len(hf_results['models'])} models, {len(hf_results['datasets'])} datasets")
            except Exception as e:
                self.logger.error(f"✗ HuggingFace failed: {e}")
        
        # OpenRouter
        if 'openrouter' in self.sources:
            try:
                self.logger.info("Fetching OpenRouter...")
                fetcher = OpenRouterFetcher()
                models = fetcher.fetch_and_normalize_all()
                
                results['models'].extend(models)
                
                self.logger.info(f"✓ OpenRouter: {len(models)} models")
            except Exception as e:
                self.logger.error(f"✗ OpenRouter failed: {e}")
        
        # GitHub
        if 'github' in self.sources:
            try:
                self.logger.info("Fetching GitHub...")
                fetcher = GitHubFetcher()
                repos = fetcher.fetch_and_normalize_all()
                
                results['repositories'].extend(repos)
                
                self.logger.info(f"✓ GitHub: {len(repos)} repositories")
            except Exception as e:
                self.logger.error(f"✗ GitHub failed: {e}")
        
        # Kaggle
        if 'kaggle' in self.sources:
            try:
                self.logger.info("Fetching Kaggle...")
                fetcher = KaggleFetcher()
                datasets = fetcher.fetch_and_normalize_all(max_pages=2)
                
                results['datasets'].extend(datasets)
                
                self.logger.info(f"✓ Kaggle: {len(datasets)} datasets")
            except ImportError:
                self.logger.warning("⚠ Kaggle not installed, skipping")
            except Exception as e:
                self.logger.error(f"✗ Kaggle failed: {e}")
        
        self.stats['models_count'] = len(results['models'])
        self.stats['datasets_count'] = len(results['datasets'])
        self.stats['repos_count'] = len(results['repositories'])
        self.stats['fetched_count'] = sum([
            len(results['models']),
            len(results['datasets']),
            len(results['repositories'])
        ])
        
        return results
    
    def _deduplicate(self, resources: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Deduplicate resources"""
        self.logger.info("\nDeduplicating...")
        self.logger.info("-" * 70)
        
        for resource_type in ['models', 'datasets', 'repositories']:
            before = len(resources[resource_type])
            
            # Deduplicate by source + source_url
            seen = {}
            for item in resources[resource_type]:
                key = f"{item['source']}:{item['source_url']}"
                if key not in seen:
                    seen[key] = item
            
            resources[resource_type] = list(seen.values())
            after = len(resources[resource_type])
            
            self.logger.info(f"{resource_type}: {before} → {after} (removed {before - after})")
        
        return resources
    
    def _save_results(self, resources: Dict[str, List[Dict]]):
        """Save results to JSON files"""
        self.logger.info("\nSaving results...")
        self.logger.info("-" * 70)
        
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(exist_ok=True)
        
        # Save models
        models_file = output_dir / 'models.json'
        with open(models_file, 'w', encoding='utf-8') as f:
            json.dump(resources['models'], f, indent=2, ensure_ascii=False)
        self.logger.info(f"✓ Saved {len(resources['models'])} models to {models_file}")
        
        # Save HuggingFace datasets
        hf_datasets = [d for d in resources['datasets'] if d['source'] == 'huggingface']
        hf_file = output_dir / 'huggingface_datasets.json'
        with open(hf_file, 'w', encoding='utf-8') as f:
            json.dump(hf_datasets, f, indent=2, ensure_ascii=False)
        self.logger.info(f"✓ Saved {len(hf_datasets)} HF datasets to {hf_file}")
        
        # Save Kaggle datasets
        kaggle_datasets = [d for d in resources['datasets'] if d['source'] == 'kaggle']
        kaggle_file = output_dir / 'kaggle_datasets.json'
        with open(kaggle_file, 'w', encoding='utf-8') as f:
            json.dump(kaggle_datasets, f, indent=2, ensure_ascii=False)
        self.logger.info(f"✓ Saved {len(kaggle_datasets)} Kaggle datasets to {kaggle_file}")
        
        # Save repositories
        repos_file = output_dir / 'github_resources.json'
        with open(repos_file, 'w', encoding='utf-8') as f:
            json.dump(resources['repositories'], f, indent=2, ensure_ascii=False)
        self.logger.info(f"✓ Saved {len(resources['repositories'])} repos to {repos_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ingestion pipeline')
    parser.add_argument('--sources', nargs='+', help='Sources to test')
    parser.add_argument('--run-id', help='Run ID')
    
    args = parser.parse_args()
    
    orchestrator = TestOrchestrator(
        run_id=args.run_id,
        sources=args.sources
    )
    
    result = orchestrator.run()
    
    if result['status'] == 'failed':
        sys.exit(1)


if __name__ == '__main__':
    main()
