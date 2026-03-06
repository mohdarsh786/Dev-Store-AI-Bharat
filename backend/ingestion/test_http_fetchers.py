"""
Test HTTP-based fetchers

Tests direct API calls to HuggingFace, OpenRouter, GitHub, and Kaggle
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fetchers.huggingface_fetcher import HuggingFaceFetcher
from fetchers.openrouter_fetcher import OpenRouterFetcher
from fetchers.github_fetcher import GitHubFetcher
from fetchers.kaggle_fetcher import KaggleFetcher


def test_huggingface():
    """Test HuggingFace API fetcher"""
    print("\n" + "=" * 60)
    print("TESTING HUGGINGFACE FETCHER")
    print("=" * 60)
    
    try:
        fetcher = HuggingFaceFetcher()
        
        # Test fetching models for one task
        print("\n1. Fetching text-classification models...")
        models = fetcher.fetch_models_by_task('text-classification', limit=5)
        print(f"   ✓ Fetched {len(models)} models")
        if models:
            print(f"   Sample: {models[0].get('modelId') or models[0].get('id')}")
        
        # Test fetching datasets
        print("\n2. Fetching datasets...")
        datasets = fetcher.fetch_datasets(limit=5)
        print(f"   ✓ Fetched {len(datasets)} datasets")
        if datasets:
            print(f"   Sample: {datasets[0].get('id')}")
        
        # Test normalization
        print("\n3. Testing normalization...")
        if models:
            normalized = fetcher.normalize_model(models[0])
            if normalized:
                print(f"   ✓ Normalized model: {normalized['name']}")
                print(f"   Source: {normalized['source']}")
                print(f"   Category: {normalized['category']}")
        
        print("\n✅ HuggingFace fetcher test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ HuggingFace fetcher test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openrouter():
    """Test OpenRouter API fetcher"""
    print("\n" + "=" * 60)
    print("TESTING OPENROUTER FETCHER")
    print("=" * 60)
    
    try:
        fetcher = OpenRouterFetcher()
        
        # Test fetching models
        print("\n1. Fetching models...")
        models = fetcher.fetch_models()
        print(f"   ✓ Fetched {len(models)} models")
        if models:
            print(f"   Sample: {models[0].get('id')}")
        
        # Test normalization
        print("\n2. Testing normalization...")
        if models:
            normalized = fetcher.normalize_model(models[0])
            if normalized:
                print(f"   ✓ Normalized model: {normalized['name']}")
                print(f"   Source: {normalized['source']}")
                print(f"   Context length: {normalized['metadata']['context_length']}")
        
        print("\n✅ OpenRouter fetcher test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ OpenRouter fetcher test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_github():
    """Test GitHub API fetcher"""
    print("\n" + "=" * 60)
    print("TESTING GITHUB FETCHER")
    print("=" * 60)
    
    try:
        fetcher = GitHubFetcher()
        
        # Test searching repositories
        print("\n1. Searching repositories...")
        repos = fetcher.search_repositories('api framework stars:>1000', per_page=5)
        print(f"   ✓ Fetched {len(repos)} repositories")
        if repos:
            print(f"   Sample: {repos[0].get('full_name')}")
            print(f"   Stars: {repos[0].get('stargazers_count')}")
        
        # Test normalization
        print("\n2. Testing normalization...")
        if repos:
            normalized = fetcher.normalize_repository(repos[0])
            if normalized:
                print(f"   ✓ Normalized repo: {normalized['name']}")
                print(f"   Source: {normalized['source']}")
                print(f"   Category: {normalized['category']}")
        
        print("\n✅ GitHub fetcher test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ GitHub fetcher test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kaggle():
    """Test Kaggle API fetcher"""
    print("\n" + "=" * 60)
    print("TESTING KAGGLE FETCHER")
    print("=" * 60)
    
    try:
        fetcher = KaggleFetcher()
        
        # Test fetching datasets
        print("\n1. Fetching datasets...")
        datasets = fetcher.fetch_datasets(page=1, page_size=5)
        print(f"   ✓ Fetched {len(datasets)} datasets")
        if datasets:
            print(f"   Sample: {datasets[0].get('ref')}")
        
        # Test normalization
        print("\n2. Testing normalization...")
        if datasets:
            normalized = fetcher.normalize_dataset(datasets[0])
            if normalized:
                print(f"   ✓ Normalized dataset: {normalized['name']}")
                print(f"   Source: {normalized['source']}")
                print(f"   Downloads: {normalized['downloads']}")
        
        print("\n✅ Kaggle fetcher test PASSED")
        return True
        
    except ImportError as e:
        print(f"\n⚠️  Kaggle fetcher not available: {e}")
        print("   Install with: pip install kaggle")
        return None  # Skip test
        
    except Exception as e:
        print(f"\n❌ Kaggle fetcher test FAILED: {e}")
        print("   Make sure Kaggle API credentials are set up")
        print("   See: https://www.kaggle.com/docs/api")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("HTTP FETCHERS TEST SUITE")
    print("=" * 60)
    print("\nTesting direct HTTP API calls")
    print("No authentication required (except GitHub optional)")
    
    results = {
        'huggingface': test_huggingface(),
        'openrouter': test_openrouter(),
        'github': test_github(),
        'kaggle': test_kaggle(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    # Count passed/failed/skipped
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)
    
    for name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
        print(f"{name.upper()}: {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped (out of {total})")
    
    if failed == 0:
        print("\n🎉 All available tests passed! HTTP fetchers are working correctly.")
        print("\nNext steps:")
        print("  1. Run full ingestion: python backend/ingestion/run_ingestion.py")
        print("  2. Check output files in: backend/ingestion/data/")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
