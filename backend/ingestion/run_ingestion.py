"""
Main Ingestion Runner

Fetches data from multiple sources and saves to JSON files:
- models.json: HuggingFace + OpenRouter models (deduplicated)
- huggingface_datasets.json: HuggingFace datasets only
- kaggle_datasets.json: Kaggle datasets only
- github_resources.json: GitHub repositories
"""
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fetchers.huggingface_fetcher import HuggingFaceFetcher
from fetchers.openrouter_fetcher import OpenRouterFetcher
from fetchers.github_fetcher import GitHubFetcher
from fetchers.kaggle_fetcher import KaggleFetcher


def save_to_json(data: dict, filename: str):
    """Save data to JSON file"""
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved to: {filepath}")
    return filepath


def deduplicate_models(hf_models: list, or_models: list) -> list:
    """
    Deduplicate models from HuggingFace and OpenRouter
    
    Deduplication strategy:
    - Use model name as key
    - If duplicate found, keep the one with more downloads/stars
    """
    models_dict = {}
    
    # Add all models to dict with name as key
    for model in hf_models + or_models:
        name = model['name'].lower()
        
        if name not in models_dict:
            models_dict[name] = model
        else:
            # Keep the one with more downloads/stars
            existing = models_dict[name]
            existing_score = existing.get('downloads', 0) + existing.get('stars', 0)
            new_score = model.get('downloads', 0) + model.get('stars', 0)
            
            if new_score > existing_score:
                models_dict[name] = model
    
    return list(models_dict.values())


def main():
    """Main ingestion pipeline"""
    print("\n" + "=" * 70)
    print("DEV STORE INGESTION PIPELINE")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    print("Output structure:")
    print("  • models.json (HuggingFace + OpenRouter, deduplicated)")
    print("  • huggingface_datasets.json")
    print("  • kaggle_datasets.json")
    print("  • github_resources.json")
    print("=" * 70)
    
    all_models = []
    
    # 1. Fetch Models from HuggingFace
    print("\n" + "=" * 70)
    print("1. HUGGINGFACE MODELS")
    print("=" * 70)
    hf_models = []
    try:
        hf_fetcher = HuggingFaceFetcher()
        hf_results = hf_fetcher.fetch_and_normalize_all()
        hf_models = hf_results['models']
        print(f"✅ Fetched {len(hf_models)} models from HuggingFace")
    except Exception as e:
        print(f"❌ HuggingFace models fetch failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Fetch Models from OpenRouter
    print("\n" + "=" * 70)
    print("2. OPENROUTER MODELS")
    print("=" * 70)
    or_models = []
    try:
        or_fetcher = OpenRouterFetcher()
        or_models = or_fetcher.fetch_and_normalize_all()
        print(f"✅ Fetched {len(or_models)} models from OpenRouter")
    except Exception as e:
        print(f"❌ OpenRouter models fetch failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Deduplicate and Save Models
    print("\n" + "=" * 70)
    print("DEDUPLICATING MODELS")
    print("=" * 70)
    all_models = deduplicate_models(hf_models, or_models)
    print(f"Total models before deduplication: {len(hf_models) + len(or_models)}")
    print(f"Total models after deduplication: {len(all_models)}")
    print(f"Duplicates removed: {len(hf_models) + len(or_models) - len(all_models)}")
    
    save_to_json(all_models, 'models.json')

    
    # 4. Fetch Datasets from HuggingFace
    print("\n" + "=" * 70)
    print("3. HUGGINGFACE DATASETS")
    print("=" * 70)
    try:
        hf_fetcher = HuggingFaceFetcher()
        hf_results = hf_fetcher.fetch_and_normalize_all()
        hf_datasets = hf_results['datasets']
        print(f"✅ Fetched {len(hf_datasets)} datasets from HuggingFace")
        save_to_json(hf_datasets, 'huggingface_datasets.json')
    except Exception as e:
        print(f"❌ HuggingFace datasets fetch failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Fetch Datasets from Kaggle
    print("\n" + "=" * 70)
    print("4. KAGGLE DATASETS")
    print("=" * 70)
    try:
        kaggle_fetcher = KaggleFetcher()
        kaggle_datasets = kaggle_fetcher.fetch_and_normalize_all(max_pages=5)
        print(f"✅ Fetched {len(kaggle_datasets)} datasets from Kaggle")
        save_to_json(kaggle_datasets, 'kaggle_datasets.json')
    except ImportError as e:
        print(f"⚠️  Kaggle package not installed: {e}")
        print("   Install with: pip install kaggle")
        print("   Skipping Kaggle datasets...")
    except Exception as e:
        print(f"❌ Kaggle fetch failed: {e}")
        print("   Make sure Kaggle API credentials are set up")
        print("   See: https://www.kaggle.com/docs/api")
        import traceback
        traceback.print_exc()
    
    # 6. Fetch Repositories from GitHub
    print("\n" + "=" * 70)
    print("5. GITHUB REPOSITORIES")
    print("=" * 70)
    try:
        gh_fetcher = GitHubFetcher()
        gh_repos = gh_fetcher.fetch_and_normalize_all()
        print(f"✅ Fetched {len(gh_repos)} repositories from GitHub")
        save_to_json(gh_repos, 'github_resources.json')
    except Exception as e:
        print(f"❌ GitHub fetch failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"Completed at: {datetime.now().isoformat()}")
    print()
    print("Output files in: backend/ingestion/output/")
    print(f"  ✓ models.json ({len(all_models)} models, deduplicated)")
    print("  ✓ huggingface_datasets.json")
    print("  ✓ kaggle_datasets.json")
    print("  ✓ github_resources.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
