"""
GitHub Resource Fetcher

Uses direct HTTP requests to fetch repositories from GitHub API
"""
import httpx
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class GitHubFetcher:
    """Fetcher for GitHub repositories using direct API calls"""
    
    # Search queries for different resource types
    SEARCH_QUERIES = [
        'api framework stars:>1000',
        'machine learning model stars:>500',
        'rest api stars:>1000',
        'graphql api stars:>500',
        'ml framework stars:>1000',
        'deep learning stars:>1000',
        'nlp library stars:>500',
        'computer vision stars:>500',
        'data science tool stars:>500',
    ]
    
    def __init__(self, token: str = None):
        # Optional token for higher rate limits (60 -> 5000 req/hour)
        self.api_token = token or os.getenv('INGESTION_GITHUB_API_TOKEN')
        self.base_url = 'https://api.github.com'
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'DevStore-Fetcher/1.0'
        }
        if self.api_token:
            self.headers['Authorization'] = f'token {self.api_token}'
    
    def search_repositories(self, query: str, per_page: int = 100) -> List[Dict[str, Any]]:
        """Search GitHub repositories"""
        url = f'{self.base_url}/search/repositories'
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': per_page
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get('items', [])
    
    def fetch_all_repositories(self, per_page: int = 100) -> List[Dict[str, Any]]:
        """Fetch repositories for all search queries"""
        all_repos = []
        seen_ids = set()
        
        for query in self.SEARCH_QUERIES:
            try:
                repos = self.search_repositories(query, per_page)
                for repo in repos:
                    repo_id = repo.get('id')
                    if repo_id and repo_id not in seen_ids:
                        seen_ids.add(repo_id)
                        all_repos.append(repo)
                print(f"✓ Fetched {len(repos)} repos for query: {query}")
            except Exception as e:
                print(f"✗ Error fetching repos for query '{query}': {e}")
        
        return all_repos
    
    def determine_category(self, repo: Dict[str, Any]) -> str:
        """Determine resource category based on topics and description"""
        topics = [t.lower() for t in repo.get('topics', [])]
        description = (repo.get('description') or '').lower()
        name = repo['name'].lower()
        
        # Check for ML models
        ml_keywords = ['model', 'neural', 'deep-learning', 'machine-learning', 'pytorch', 'tensorflow', 'transformers']
        if any(kw in topics or kw in description or kw in name for kw in ml_keywords):
            return 'model'
        
        # Check for datasets
        dataset_keywords = ['dataset', 'data', 'corpus', 'benchmark']
        if any(kw in topics or kw in description or kw in name for kw in dataset_keywords):
            return 'dataset'
        
        # Check for APIs
        api_keywords = ['api', 'rest', 'graphql', 'sdk', 'client', 'wrapper']
        if any(kw in topics or kw in description or kw in name for kw in api_keywords):
            return 'api'
        
        # Default to solution
        return 'solution'
    
    def normalize_repository(self, repo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize repository data to standard format"""
        try:
            category = self.determine_category(repo)
            
            tags = repo.get('topics', [])
            if repo.get('language'):
                tags.append(repo['language'].lower())
            
            license_name = 'Unknown'
            if repo.get('license'):
                license_name = repo['license'].get('name', 'Unknown')
            
            return {
                'name': repo['name'],
                'description': repo.get('description', ''),
                'source': 'github',
                'source_url': repo['html_url'],
                'author': repo['owner']['login'],
                'stars': repo.get('stargazers_count', 0),
                'downloads': 0,
                'license': license_name,
                'tags': tags[:10],
                'version': repo.get('default_branch', 'main'),
                'category': category,
                'thumbnail_url': repo['owner'].get('avatar_url'),
                'readme_url': f"{repo['html_url']}/blob/{repo.get('default_branch', 'main')}/README.md",
                'metadata': {
                    'language': repo.get('language'),
                    'forks': repo.get('forks_count', 0),
                    'watchers': repo.get('watchers_count', 0),
                    'open_issues': repo.get('open_issues_count', 0),
                    'created_at': repo.get('created_at'),
                    'updated_at': repo.get('updated_at'),
                    'pushed_at': repo.get('pushed_at'),
                    'size': repo.get('size', 0),
                    'has_wiki': repo.get('has_wiki', False),
                    'has_pages': repo.get('has_pages', False),
                },
                'scraped_at': datetime.utcnow().isoformat()
            }
        except Exception as e:
            print(f"Error normalizing repo {repo.get('html_url')}: {e}")
            return None
    
    def fetch_and_normalize_all(self) -> List[Dict[str, Any]]:
        """Fetch and normalize all repositories"""
        print("Fetching GitHub repositories...")
        raw_repos = self.fetch_all_repositories()
        
        repos = [self.normalize_repository(r) for r in raw_repos]
        repos = [r for r in repos if r is not None]
        
        print(f"\n✅ Total fetched: {len(repos)} repositories")
        
        return repos


if __name__ == '__main__':
    # Test the fetcher
    fetcher = GitHubFetcher()
    repos = fetcher.fetch_and_normalize_all()
    
    print(f"\nSample repo: {repos[0]['name']}")
    print(f"Stars: {repos[0]['stars']}")
    print(f"Category: {repos[0]['category']}")
