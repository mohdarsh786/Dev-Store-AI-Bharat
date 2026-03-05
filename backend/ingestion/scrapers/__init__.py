"""
Scrapy Spiders

Only used for web scraping when no official API is available
"""
from .rapidapi_spider import RapidAPIResourceSpider

__all__ = ['RapidAPIResourceSpider']
