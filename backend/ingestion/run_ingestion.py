"""
Main Ingestion Runner

Uses HTTP requests for HuggingFace, GitHub, and OpenRouter APIs
Uses Scrapy crawler only for RapidAPI (no official API available)
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


def save_to_json(data: dict, filename: str):
    """Save data to JSON file"""
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved to: {filepath}")


def run_http_fetchers():
    """Run all HTTP-based fetchers"""
    print("=" * 70)
    print("RUNNING HTTP-BASED FETCHERS")
    print("=" * 70)
    
    all_resources = []
    
    # 1. Fetch from HuggingFace
    print("\n" + "=" * 70)
    print("1. HUGGINGFACE API")
    print("=" * 70)
    try:
        hf_fetcher = HuggingFaceFetcher()
        hf_results = hf_fetcher.fetch_and_normalize_all()
        all_resources.extend(hf_results['models'])
        all_resources.extend(hf_results['datasets'])
        save_to_json(hf_results, 'huggingface_resources.json')
    except Exception as e:
        print(f"❌ HuggingFace fetch failed: {e}")
    
    # 2. Fetch from OpenRouter
    print("\n" + "=" * 70)
    print("2. OPENROUTER API")
    print("=" * 70)
    try:
        or_fetcher = OpenRouterFetcher()
        or_models = or_fetcher.fetch_and_normalize_all()
        all_resources.extend(or_models)
        save_to_json({'models': or_models}, 'openrouter_resources.json')
    except Exception as e:
        print(f"❌ OpenRouter fetch failed: {e}")
    
    # 3. Fetch from GitHub
    print("\n" + "=" * 70)
    print("3. GITHUB API")
    print("=" * 70)
    try:
        gh_fetcher = GitHubFetcher()
        gh_repos = gh_fetcher.fetch_and_normalize_all()
        all_resources.extend(gh_repos)
        save_to_json({'repositories': gh_repos}, 'github_resources.json')
    except Exception as e:
        print(f"❌ GitHub fetch failed: {e}")
    
    return all_resources


def run_rapidapi_crawler():
    """Run Scrapy crawler for RapidAPI (no official API available)"""
    print("\n" + "=" * 70)
    print("4. RAPIDAPI CRAWLER (Web Scraping)")
    print("=" * 70)
    print("Note: RapidAPI has no official API, using web scraping")
    
    try:
        from scrapy.crawler import CrawlerProcess
        from scrapy.utils.project import get_project_settings
        from scrapers.rapidapi_spider import RapidAPIResourceSpider
        
        # Configure Scrapy settings
        settings = get_project_settings()
        settings.update({
            'FEEDS': {
                'output/rapidapi_resources.json': {
                    'format': 'json',
                    'encoding': 'utf-8',
                    'overwrite': True,
                }
            },
            'LOG_LEVEL': 'INFO',
        })
        
        # Run crawler
        process = CrawlerProcess(settings)
        process.crawl(RapidAPIResourceSpider)
        process.start()
        
        print("✅ RapidAPI crawling completed")
        
    except Exception as e:
        print(f"❌ RapidAPI crawl failed: {e}")
        print("Make sure Scrapy is installed and scrapy.cfg is configured")


def main():
    """Main ingestion pipeline"""
    print("\n" + "=" * 70)
    print("DEV STORE INGESTION PIPELINE")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    print("Strategy:")
    print("  • HuggingFace: HTTP API requests")
    print("  • OpenRouter: HTTP API requests")
    print("  • GitHub: HTTP API requests")
    print("  • RapidAPI: Scrapy crawler (no official API)")
    print("=" * 70)
    
    # Run HTTP fetchers
    all_resources = run_http_fetchers()
    
    # Run RapidAPI crawler
    run_rapidapi_crawler()
    
    # Summary
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"Total resources fetched (HTTP): {len(all_resources)}")
    print(f"Completed at: {datetime.now().isoformat()}")
    print()
    print("Output files:")
    print("  • output/huggingface_resources.json")
    print("  • output/openrouter_resources.json")
    print("  • output/github_resources.json")
    print("  • output/rapidapi_resources.json")
    print("=" * 70)
    
    # Save combined resources
    save_to_json({
        'total_count': len(all_resources),
        'resources': all_resources,
        'fetched_at': datetime.now().isoformat()
    }, 'all_resources_combined.json')


if __name__ == '__main__':
    main()
