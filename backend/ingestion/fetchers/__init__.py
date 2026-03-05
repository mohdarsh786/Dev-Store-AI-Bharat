"""
Resource Fetchers

Direct HTTP-based fetchers for external APIs
"""
from .huggingface_fetcher import HuggingFaceFetcher
from .openrouter_fetcher import OpenRouterFetcher
from .github_fetcher import GitHubFetcher

__all__ = [
    'HuggingFaceFetcher',
    'OpenRouterFetcher',
    'GitHubFetcher',
]
