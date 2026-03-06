"""
Kaggle Dataset Fetcher

Uses Kaggle API to fetch datasets
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class KaggleFetcher:
    """Fetcher for Kaggle datasets using Kaggle API"""
    
    def __init__(self):
        """
        Initialize Kaggle fetcher
        
        Requires Kaggle API credentials:
        - Set up ~/.kaggle/kaggle.json with your API credentials
        - Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables
        
        Get credentials from: https://www.kaggle.com/settings/account
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            self.api = KaggleApi()
            self.api.authenticate()
        except ImportError:
            raise ImportError(
                "Kaggle package not installed. Install with: pip install kaggle"
            )
        except Exception as e:
            raise Exception(
                f"Kaggle authentication failed: {e}\n"
                "Make sure you have set up Kaggle API credentials.\n"
                "See: https://www.kaggle.com/docs/api"
            )
    
    def fetch_datasets(self, page: int = 1, page_size: int = 100, sort_by: str = 'hottest') -> List[Dict[str, Any]]:
        """
        Fetch datasets from Kaggle
        
        Args:
            page: Page number (1-indexed)
            page_size: Number of datasets per page (max 100)
            sort_by: Sort order ('hottest', 'votes', 'updated', 'active')
        
        Returns:
            List of dataset metadata
        """
        try:
            datasets = self.api.dataset_list(
                page=page,
                sort_by=sort_by
            )
            return [self._dataset_to_dict(d) for d in datasets]
        except Exception as e:
            print(f"Error fetching Kaggle datasets: {e}")
            return []
    
    def _dataset_to_dict(self, dataset) -> Dict[str, Any]:
        """Convert Kaggle dataset object to dictionary"""
        return {
            'ref': getattr(dataset, "ref", None),
            'title': getattr(dataset, "title", None),
            'lastUpdated': str(getattr(dataset, "lastUpdated", None)),
            'downloadCount': getattr(dataset, "downloadCount", 0),
            'voteCount': getattr(dataset, "voteCount", 0),
            'usabilityRating': getattr(dataset, "usabilityRating", 0),
        }
    
    def normalize_dataset(self, dataset: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize dataset data to standard format"""
        try:
            ref = dataset.get('ref', '')
            if not ref:
                return None
            
            # Extract owner and dataset name from ref (format: owner/dataset-name)
            parts = ref.split('/')
            owner = parts[0] if len(parts) > 0 else 'unknown'
            dataset_name = parts[1] if len(parts) > 1 else ref
            
            return {
                'name': dataset_name,
                'description': dataset.get('title', ''),
                'source': 'kaggle',
                'source_url': f"https://www.kaggle.com/datasets/{ref}",
                'author': owner,
                'stars': dataset.get('voteCount', 0),
                'downloads': dataset.get('downloadCount', 0),
                'license': 'Various',  # Kaggle datasets have different licenses
                'tags': ['kaggle', 'dataset'],
                'version': 'latest',
                'category': 'dataset',
                'thumbnail_url': None,
                'readme_url': f"https://www.kaggle.com/datasets/{ref}",
                'metadata': {
                    'ref': ref,
                    'last_updated': dataset.get('lastUpdated'),
                    'usability_rating': dataset.get('usabilityRating', 0),
                },
                'scraped_at': datetime.utcnow().isoformat()
            }
        except Exception as e:
            print(f"Error normalizing Kaggle dataset: {e}")
            return None
    
    def fetch_and_normalize_all(self, max_pages: int = 5, page_size: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch and normalize datasets from multiple pages
        
        Args:
            max_pages: Maximum number of pages to fetch
            page_size: Number of datasets per page
        
        Returns:
            List of normalized datasets
        """
        print(f"Fetching Kaggle datasets (up to {max_pages} pages)...")
        
        all_datasets = []
        for page in range(1, max_pages + 1):
            try:
                datasets = self.fetch_datasets(page=page, page_size=page_size)
                if not datasets:
                    break
                
                normalized = [self.normalize_dataset(d) for d in datasets]
                normalized = [d for d in normalized if d is not None]
                all_datasets.extend(normalized)
                
                print(f"  Page {page}: Fetched {len(normalized)} datasets")
            except Exception as e:
                print(f"  Page {page}: Error - {e}")
                break
        
        print(f"✅ Total fetched: {len(all_datasets)} datasets from Kaggle")
        return all_datasets


if __name__ == '__main__':
    # Test the fetcher
    try:
        fetcher = KaggleFetcher()
        datasets = fetcher.fetch_and_normalize_all(max_pages=2)
        
        if datasets:
            print(f"\nSample dataset: {datasets[0]['name']}")
            print(f"Downloads: {datasets[0]['downloads']}")
            print(f"Votes: {datasets[0]['stars']}")
    except ImportError as e:
        print(f"⚠️  {e}")
        print("Install with: pip install kaggle")
    except Exception as e:
        print(f"❌ Error: {e}")
