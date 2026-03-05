"""
Scrapy settings for DevStore ingestion spiders
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ingestion_settings

# Scrapy settings
BOT_NAME = 'devstore_scraper'

SPIDER_MODULES = ['scrapers']
NEWSPIDER_MODULE = 'scrapers'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = ingestion_settings.scraper_concurrent_requests

# Configure a delay for requests for the same website
DOWNLOAD_DELAY = ingestion_settings.scraper_download_delay

# Disable cookies
COOKIES_ENABLED = False

# Disable Telnet Console
TELNETCONSOLE_ENABLED = False

# Override the default request headers
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'application/json',
    'Accept-Language': 'en',
}

# Enable or disable spider middlewares
SPIDER_MIDDLEWARES = {
    'scrapy.spidermiddlewares.httperror.HttpErrorMiddleware': 50,
}

# Enable or disable downloader middlewares
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
}

# Configure item pipelines
ITEM_PIPELINES = {
    'scrapers.pipelines.DeduplicationPipeline': 100,
    'scrapers.pipelines.ValidationPipeline': 200,
    'scrapers.pipelines.SQSPipeline': 300,
}

# Enable and configure HTTP caching
HTTPCACHE_ENABLED = False

# Set settings whose default value is deprecated
REQUEST_FINGERPRINTER_IMPLEMENTATION = '2.7'
TWISTED_REACTOR = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'
FEED_EXPORT_ENCODING = 'utf-8'

# Retry settings
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Timeout settings
DOWNLOAD_TIMEOUT = 30

# AutoThrottle settings
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
