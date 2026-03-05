"""
RapidAPI Resource Spider

Scrapes APIs from RapidAPI marketplace using their search API
"""
import scrapy
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime


class RapidAPIResourceSpider(scrapy.Spider):
    """
    Spider for scraping APIs from RapidAPI marketplace.
    
    Scrapes:
    - Public APIs
    - API metadata and pricing
    - Popularity metrics
    """
    
    name = 'rapidapi_resource'
    allowed_domains = ['rapidapi.com']
    
    # Popular API categories
    CATEGORIES = [
        'artificial-intelligence',
        'machine-learning',
        'data',
        'text-analysis',
        'image-processing',
        'video',
        'translation',
        'weather',
        'finance',
        'social',
        'sports',
        'news',
        'entertainment',
        'tools',
    ]
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1.0,
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get API key from environment variable (optional)
        self.api_key = os.getenv('INGESTION_RAPIDAPI_KEY')
        self.base_url = 'https://rapidapi.com'
        self.headers = {
            'Accept': 'application/json',
            'User-Agent': 'DevStore-Scraper/1.0'
        }
        if self.api_key:
            self.headers['X-RapidAPI-Key'] = self.api_key
    
    def start_requests(self):
        """Generate initial requests for API categories"""
        # Use RapidAPI's search endpoint for each category
        for category in self.CATEGORIES:
            # Search for APIs in this category
            url = f'{self.base_url}/search/{category}'
            yield scrapy.Request(
                url=url,
                headers=self.headers,
                callback=self.parse_search_results,
                meta={'category': category},
                dont_filter=True
            )
    
    def parse_search_results(self, response):
        """Parse search results and extract API links"""
        try:
            # Try to extract JSON data from the page
            # RapidAPI often embeds data in script tags
            json_data = response.css('script[type="application/json"]::text').get()
            
            if json_data:
                data = json.loads(json_data)
                apis = data.get('apis', []) or data.get('results', [])
                
                for api in apis:
                    resource = self.extract_api_from_json(api, response.meta['category'])
                    if resource:
                        yield resource
            else:
                # Fallback to HTML parsing
                api_cards = response.css('div.api-card, article.api-item')
                
                for card in api_cards:
                    api_link = card.css('a::attr(href)').get()
                    if api_link:
                        if not api_link.startswith('http'):
                            api_link = f'{self.base_url}{api_link}'
                        
                        yield scrapy.Request(
                            url=api_link,
                            headers=self.headers,
                            callback=self.parse_api_detail,
                            meta={'category': response.meta['category']}
                        )
                        
        except Exception as e:
            self.logger.error(f"Error parsing search results: {e}")
    
    def extract_api_from_json(self, api_data: Dict[str, Any], category: str) -> Optional[Dict[str, Any]]:
        """Extract API resource from JSON data"""
        try:
            api_name = api_data.get('name') or api_data.get('title')
            if not api_name:
                return None
            
            api_id = api_data.get('id') or api_data.get('slug', '')
            api_url = f"{self.base_url}/api/{api_id}" if api_id else None
            
            # Extract pricing
            pricing = api_data.get('pricing', {})
            has_free = pricing.get('free', False) or any(
                plan.get('price', 0) == 0 for plan in pricing.get('plans', [])
            )
            pricing_type = 'free' if has_free else 'freemium'
            
            # Build tags
            tags = [category]
            if api_data.get('category'):
                tags.append(api_data['category'])
            if api_data.get('tags'):
                tags.extend(api_data['tags'][:5])
            
            resource = {
                'name': api_name,
                'description': api_data.get('description', ''),
                'source': 'rapidapi',
                'source_url': api_url or f"{self.base_url}/search/{api_name}",
                'author': api_data.get('provider', {}).get('name', 'Unknown'),
                'stars': api_data.get('rating', 0),
                'downloads': api_data.get('popularity', 0),
                'license': 'Commercial',
                'tags': tags[:10],
                'version': api_data.get('version', '1.0'),
                'category': 'api',
                'thumbnail_url': api_data.get('image') or api_data.get('thumbnail'),
                'readme_url': api_url,
                'metadata': {
                    'pricing_type': pricing_type,
                    'rating': api_data.get('rating', 0),
                    'latency': api_data.get('latency'),
                    'popularity': api_data.get('popularity', 0),
                    'category': category,
                    'endpoints': api_data.get('endpoints', []),
                },
                'scraped_at': datetime.utcnow().isoformat()
            }
            
            return resource
            
        except Exception as e:
            self.logger.error(f"Error extracting API from JSON: {e}")
            return None
    
    def parse_api_detail(self, response):
        """Parse API detail page and extract API data"""
        try:
            # Extract API data from page
            # Note: This is a simplified version. Real implementation would need
            # to parse the actual HTML structure or use RapidAPI's API
            
            api_name = response.css('h1.api-title::text').get()
            if not api_name:
                return
            
            description = response.css('div.api-description::text').get() or ''
            author = response.css('span.api-author::text').get() or 'Unknown'
            rating = response.css('span.api-rating::text').get() or '0'
            
            # Extract pricing info
            pricing_type = 'freemium'  # Most RapidAPI APIs are freemium
            if 'free' in description.lower():
                pricing_type = 'free'
            elif 'paid' in description.lower() or 'premium' in description.lower():
                pricing_type = 'paid'
            
            # Build resource object
            resource = {
                'name': api_name.strip(),
                'description': description.strip(),
                'source': 'rapidapi',
                'source_url': response.url,
                'author': author.strip(),
                'stars': 0,  # RapidAPI doesn't use stars
                'downloads': 0,  # Not publicly available
                'license': 'Commercial',
                'tags': [response.meta['category']],
                'version': '1.0',
                'category': 'api',
                'thumbnail_url': response.css('img.api-logo::attr(src)').get(),
                'readme_url': response.url,
                'metadata': {
                    'pricing_type': pricing_type,
                    'rating': float(rating) if rating.replace('.', '').isdigit() else 0.0,
                    'category': response.meta['category'],
                },
                'scraped_at': datetime.utcnow().isoformat()
            }
            
            yield resource
            
        except Exception as e:
            self.logger.error(f"Error parsing API detail from {response.url}: {e}")
