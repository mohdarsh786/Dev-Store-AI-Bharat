"""
OpenRouter Resource Fetcher

Uses direct HTTP requests to fetch LLM models from OpenRouter API
"""
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime


class OpenRouterFetcher:
    """Fetcher for OpenRouter models using direct API calls"""
    
    def __init__(self):
        # No API key required for public models endpoint
        self.base_url = 'https://openrouter.ai/api/v1'
        self.headers = {
            'Accept': 'application/json',
            'User-Agent': 'DevStore-Fetcher/1.0'
        }
    
    def fetch_models(self) -> List[Dict[str, Any]]:
        """Fetch all available models from OpenRouter"""
        url = f'{self.base_url}/models'
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
    
    def normalize_model(self, model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize model data to standard format"""
        model_id = model.get('id')
        if not model_id:
            return None
        
        # Extract model name and provider
        name_parts = model_id.split('/')
        provider = name_parts[0] if len(name_parts) > 1 else 'openrouter'
        model_name = name_parts[-1]
        
        # Extract pricing info
        pricing = model.get('pricing', {})
        prompt_price = pricing.get('prompt', '0')
        completion_price = pricing.get('completion', '0')
        
        # Build tags from model capabilities
        tags = []
        if model.get('architecture'):
            tags.append(model['architecture'].get('modality', 'text'))
        if model.get('top_provider'):
            tags.append(f"provider:{model['top_provider'].get('name', provider)}")
        
        # Determine pricing type
        pricing_type = 'free' if float(prompt_price) == 0 and float(completion_price) == 0 else 'paid'
        
        return {
            'name': model_name,
            'description': model.get('description', f"OpenRouter model: {model_id}"),
            'source': 'openrouter',
            'source_url': f"https://openrouter.ai/models/{model_id}",
            'author': provider,
            'stars': 0,
            'downloads': 0,
            'license': 'Commercial',
            'tags': tags[:10],
            'version': '1.0',
            'category': 'model',
            'thumbnail_url': None,
            'readme_url': f"https://openrouter.ai/models/{model_id}",
            'metadata': {
                'model_id': model_id,
                'context_length': model.get('context_length', 0),
                'pricing': {
                    'prompt': prompt_price,
                    'completion': completion_price,
                    'currency': 'USD'
                },
                'pricing_type': pricing_type,
                'architecture': model.get('architecture', {}),
                'top_provider': model.get('top_provider', {}),
                'per_request_limits': model.get('per_request_limits'),
            },
            'scraped_at': datetime.utcnow().isoformat()
        }
    
    def fetch_and_normalize_all(self) -> List[Dict[str, Any]]:
        """Fetch and normalize all models"""
        print("Fetching OpenRouter models...")
        raw_models = self.fetch_models()
        
        models = [self.normalize_model(m) for m in raw_models]
        models = [m for m in models if m is not None]
        
        print(f"✅ Total fetched: {len(models)} models")
        
        return models


if __name__ == '__main__':
    # Test the fetcher
    fetcher = OpenRouterFetcher()
    models = fetcher.fetch_and_normalize_all()
    
    print(f"\nSample model: {models[0]['name']}")
    print(f"Context length: {models[0]['metadata']['context_length']}")
    print(f"Pricing: ${models[0]['metadata']['pricing']['prompt']}/token")
