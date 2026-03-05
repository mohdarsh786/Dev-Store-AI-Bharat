"""
Test HTTP-based fetchers

Tests direct API calls to HuggingFace, OpenRouter, and GitHub
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fetchers.huggingface_fetcher import HuggingFaceFetcher
from fetchers.openrouter_fetcher import OpenRouterFetcher
from fetchers.github_fetcher import GitHubFetcher


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


def main():
    """Run all tests"""
    print("=" * 60)
    print("HTTP FETCHERS TEST SUITE")
    print("=" * 60)
    print("\nTesting direct HTTP API calls (no Scrapy)")
    print("No authentication required for any API")
    
    results = {
        'huggingface': test_huggingface(),
        'openrouter': test_openrouter(),
        'github': test_github(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name.upper()}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! HTTP fetchers are working correctly.")
        print("\nNext steps:")
        print("  1. Run full ingestion: python backend/ingestion/run_ingestion.py")
        print("  2. Check output files in: backend/ingestion/output/")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
