"""
HuggingFace Resource Fetcher

Uses direct HTTP requests to fetch ML models and datasets from HuggingFace API
"""
import httpx
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class HuggingFaceFetcher:
    """Fetcher for HuggingFace models and datasets using direct API calls"""
    
    # Popular tasks to fetch
    TASKS = [
        'text-classification',
        'token-classification',
        'question-answering',
        'summarization',
        'translation',
        'text-generation',
        'image-classification',
        'object-detection',
        'image-segmentation',
        'text-to-image',
        'automatic-speech-recognition',
    ]
    
    def __init__(self):
        # No API token required - public API
        self.base_url = 'https://huggingface.co/api'
        self.headers = {
            'Accept': 'application/json',
            'User-Agent': 'DevStore-Fetcher/1.0'
        }
    
    def fetch_models_by_task(self, task: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch models for a specific task"""
        url = f'{self.base_url}/models'
        params = {
            'pipeline_tag': task,
            'sort': 'downloads',
            'direction': -1,
            'limit': limit
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
    
    def fetch_all_models(self, limit_per_task: int = 100) -> List[Dict[str, Any]]:
        """Fetch models for all tasks"""
        all_models = []
        seen_ids = set()
        
        for task in self.TASKS:
            try:
                models = self.fetch_models_by_task(task, limit_per_task)
                for model in models:
                    model_id = model.get('modelId') or model.get('id')
                    if model_id and model_id not in seen_ids:
                        seen_ids.add(model_id)
                        all_models.append(model)
                print(f"✓ Fetched {len(models)} models for task: {task}")
            except Exception as e:
                print(f"✗ Error fetching models for task {task}: {e}")
        
        return all_models
    
    def fetch_datasets(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch popular datasets"""
        url = f'{self.base_url}/datasets'
        params = {
            'sort': 'downloads',
            'direction': -1,
            'limit': limit
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
    
    def normalize_model(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize model data to standard format"""
        model_id = model.get('modelId') or model.get('id')
        if not model_id:
            return None
        
        tags = model.get('tags', [])
        if model.get('pipeline_tag'):
            tags.append(model['pipeline_tag'])
        
        return {
            'name': model_id.split('/')[-1],
            'description': model.get('description', '') or f"HuggingFace model: {model_id}",
            'source': 'huggingface',
            'source_url': f"https://huggingface.co/{model_id}",
            'author': model_id.split('/')[0] if '/' in model_id else 'huggingface',
            'stars': model.get('likes', 0),
            'downloads': model.get('downloads', 0),
            'license': model.get('license', 'Unknown'),
            'tags': tags[:10],
            'version': model.get('sha', 'main'),
            'category': 'model',
            'thumbnail_url': None,
            'readme_url': f"https://huggingface.co/{model_id}/blob/main/README.md",
            'metadata': {
                'model_id': model_id,
                'pipeline_tag': model.get('pipeline_tag'),
                'library_name': model.get('library_name'),
                'created_at': model.get('createdAt'),
                'last_modified': model.get('lastModified'),
                'private': model.get('private', False),
                'gated': model.get('gated', False),
            },
            'scraped_at': datetime.utcnow().isoformat()
        }
    
    def normalize_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize dataset data to standard format"""
        dataset_id = dataset.get('id')
        if not dataset_id:
            return None
        
        tags = dataset.get('tags', [])
        
        return {
            'name': dataset_id.split('/')[-1],
            'description': dataset.get('description', '') or f"HuggingFace dataset: {dataset_id}",
            'source': 'huggingface',
            'source_url': f"https://huggingface.co/datasets/{dataset_id}",
            'author': dataset_id.split('/')[0] if '/' in dataset_id else 'huggingface',
            'stars': dataset.get('likes', 0),
            'downloads': dataset.get('downloads', 0),
            'license': dataset.get('license', 'Unknown'),
            'tags': tags[:10],
            'version': dataset.get('sha', 'main'),
            'category': 'dataset',
            'thumbnail_url': None,
            'readme_url': f"https://huggingface.co/datasets/{dataset_id}/blob/main/README.md",
            'metadata': {
                'dataset_id': dataset_id,
                'created_at': dataset.get('createdAt'),
                'last_modified': dataset.get('lastModified'),
                'private': dataset.get('private', False),
                'gated': dataset.get('gated', False),
            },
            'scraped_at': datetime.utcnow().isoformat()
        }
    
    def fetch_and_normalize_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch and normalize all resources"""
        print("Fetching HuggingFace models...")
        raw_models = self.fetch_all_models()
        models = [self.normalize_model(m) for m in raw_models]
        models = [m for m in models if m is not None]
        
        print(f"\nFetching HuggingFace datasets...")
        raw_datasets = self.fetch_datasets()
        datasets = [self.normalize_dataset(d) for d in raw_datasets]
        datasets = [d for d in datasets if d is not None]
        
        print(f"\n✅ Total fetched: {len(models)} models, {len(datasets)} datasets")
        
        return {
            'models': models,
            'datasets': datasets
        }


if __name__ == '__main__':
    # Test the fetcher
    fetcher = HuggingFaceFetcher()
    results = fetcher.fetch_and_normalize_all()
    
    print(f"\nSample model: {results['models'][0]['name']}")
    print(f"Sample dataset: {results['datasets'][0]['name']}")
