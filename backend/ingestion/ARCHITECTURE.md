# Ingestion Pipeline Architecture

## Overview

The Dev Store ingestion pipeline fetches ML models, datasets, APIs, and tools from multiple external sources using two different approaches:

1. **HTTP API Requests** - For sources with official REST APIs (HuggingFace, OpenRouter, GitHub)
2. **Web Scraping** - For sources without official APIs (RapidAPI)

## Design Decision: HTTP vs Scrapy

### Why HTTP for Most Sources?

We use direct HTTP requests (`httpx` library) instead of Scrapy for HuggingFace, OpenRouter, and GitHub because:

1. **Simplicity**: Direct API calls are simpler and more maintainable
2. **Performance**: Faster for REST APIs (no Scrapy overhead)
3. **Error Handling**: Better control over retries and rate limiting
4. **Debugging**: Easier to debug and test
5. **Dependencies**: Lighter weight (just `httpx` vs full Scrapy)

### When to Use Scrapy?

We only use Scrapy for RapidAPI because:
- No official REST API available
- Need to scrape HTML pages
- Scrapy excels at web scraping with built-in features:
  - Automatic retries
  - Concurrent requests
  - HTML parsing
  - Middleware support

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ├─────────────────────────────────┐
                              │                                 │
                    ┌─────────▼─────────┐          ┌───────────▼──────────┐
                    │  HTTP FETCHERS    │          │  SCRAPY CRAWLER      │
                    │  (Direct API)     │          │  (Web Scraping)      │
                    └─────────┬─────────┘          └───────────┬──────────┘
                              │                                 │
        ┌─────────────────────┼─────────────────────┐          │
        │                     │                     │          │
┌───────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐ │
│ HuggingFace    │  │  OpenRouter     │  │    GitHub       │ │
│ API Fetcher    │  │  API Fetcher    │  │  API Fetcher    │ │
│                │  │                 │  │                 │ │
│ • Models       │  │ • LLM Models    │  │ • Repositories  │ │
│ • Datasets     │  │ • Pricing       │  │ • Tools         │ │
└────────────────┘  └─────────────────┘  └─────────────────┘ │
                                                               │
                                                    ┌──────────▼──────────┐
                                                    │   RapidAPI Spider   │
                                                    │                     │
                                                    │ • API Marketplace   │
                                                    │ • Web Scraping      │
                                                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Normalization  │
                    │  & Validation   │
                    └─────────┬───────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  JSON Output    │
                    │  Files          │
                    └─────────────────┘
```

## Components

### 1. HTTP Fetchers (`fetchers/`)

Direct API clients using `httpx` library:

#### HuggingFace Fetcher
- **File**: `fetchers/huggingface_fetcher.py`
- **API**: `https://huggingface.co/api`
- **Auth**: Optional (increases rate limit)
- **Fetches**:
  - Models by task (text-classification, image-classification, etc.)
  - Datasets sorted by downloads
- **Rate Limit**: 1000 req/hour (5000 with token)

#### OpenRouter Fetcher
- **File**: `fetchers/openrouter_fetcher.py`
- **API**: `https://openrouter.ai/api/v1`
- **Auth**: Not required
- **Fetches**:
  - All available LLM models
  - Pricing information
  - Context windows
- **Rate Limit**: No documented limit

#### GitHub Fetcher
- **File**: `fetchers/github_fetcher.py`
- **API**: `https://api.github.com`
- **Auth**: Optional (increases rate limit)
- **Fetches**:
  - Repositories by search queries
  - Stars, forks, metadata
- **Rate Limit**: 60 req/hour (5000 with token)

### 2. Scrapy Crawler (`scrapers/`)

Web scraping for sites without official APIs:

#### RapidAPI Spider
- **File**: `scrapers/rapidapi_spider.py`
- **Method**: Web scraping (HTML parsing)
- **Why**: No official REST API available
- **Fetches**:
  - API marketplace listings
  - API metadata and pricing
  - Categories and ratings

### 3. Data Normalization

All fetchers normalize data to a standard format:

```python
{
    'name': str,              # Resource name
    'description': str,       # Description
    'source': str,           # Source platform (huggingface, github, etc.)
    'source_url': str,       # Original URL
    'author': str,           # Creator/organization
    'stars': int,            # Popularity metric
    'downloads': int,        # Download count
    'license': str,          # License type
    'tags': List[str],       # Tags/topics (max 10)
    'version': str,          # Version identifier
    'category': str,         # model/dataset/api/solution
    'thumbnail_url': str,    # Image URL
    'readme_url': str,       # Documentation URL
    'metadata': dict,        # Source-specific metadata
    'scraped_at': str        # ISO timestamp
}
```

## File Structure

```
backend/ingestion/
├── fetchers/                    # HTTP-based fetchers
│   ├── __init__.py
│   ├── huggingface_fetcher.py  # HuggingFace API client
│   ├── openrouter_fetcher.py   # OpenRouter API client
│   └── github_fetcher.py       # GitHub API client
│
├── scrapers/                    # Scrapy spiders
│   ├── __init__.py
│   ├── rapidapi_spider.py      # RapidAPI web scraper
│   ├── pipelines.py            # Scrapy pipelines
│   └── settings.py             # Scrapy settings
│
├── services/                    # Shared services
│   ├── api_clients.py          # Legacy async clients
│   ├── storage_service.py      # Storage utilities
│   └── sqs_service.py          # AWS SQS integration
│
├── workers/                     # Background workers
│   ├── batch_processor.py      # Batch processing
│   └── embedder.py             # Embedding generation
│
├── output/                      # Output directory (created at runtime)
│   ├── huggingface_resources.json
│   ├── openrouter_resources.json
│   ├── github_resources.json
│   ├── rapidapi_resources.json
│   └── all_resources_combined.json
│
├── run_ingestion.py            # Main runner script
├── test_http_fetchers.py       # Test HTTP fetchers
├── config.py                   # Configuration
├── scrapy.cfg                  # Scrapy configuration
├── QUICKSTART.md               # Quick start guide
└── README.md                   # Full documentation
```

## Usage

### Test HTTP Fetchers

```bash
cd backend
.venv\Scripts\python.exe ingestion\test_http_fetchers.py
```

### Run Full Ingestion

```bash
cd backend
.venv\Scripts\python.exe ingestion\run_ingestion.py
```

### Run Individual Fetchers

```python
# HuggingFace
from fetchers.huggingface_fetcher import HuggingFaceFetcher
fetcher = HuggingFaceFetcher()
results = fetcher.fetch_and_normalize_all()

# OpenRouter
from fetchers.openrouter_fetcher import OpenRouterFetcher
fetcher = OpenRouterFetcher()
models = fetcher.fetch_and_normalize_all()

# GitHub
from fetchers.github_fetcher import GitHubFetcher
fetcher = GitHubFetcher()
repos = fetcher.fetch_and_normalize_all()
```

### Run RapidAPI Crawler

```bash
cd backend/ingestion
scrapy crawl rapidapi_resource -o output/rapidapi.json
```

## Performance

| Source | Method | Time | Resources |
|--------|--------|------|-----------|
| HuggingFace | HTTP | ~30s | ~1000 models + 100 datasets |
| OpenRouter | HTTP | ~2s | ~100 models |
| GitHub | HTTP | ~2m | ~900 repositories |
| RapidAPI | Scrapy | Varies | Depends on site |

**Total**: ~3-5 minutes for all sources

## Error Handling

### HTTP Fetchers
- Automatic retries with exponential backoff
- Rate limit detection and handling
- Graceful degradation (continue on error)
- Detailed error logging

### Scrapy Crawler
- Built-in retry middleware
- Concurrent request limiting
- HTML parsing error handling
- Automatic throttling

## Rate Limits

| Source | Without Token | With Token | How to Get Token |
|--------|---------------|------------|------------------|
| HuggingFace | 1000/hour | 5000/hour | https://huggingface.co/settings/tokens |
| GitHub | 60/hour | 5000/hour | https://github.com/settings/tokens |
| OpenRouter | Unlimited | N/A | Not required |
| RapidAPI | N/A | N/A | Web scraping (no API) |

## Future Enhancements

1. **Incremental Updates**: Only fetch new/updated resources
2. **Caching**: Cache API responses to reduce API calls
3. **Deduplication**: Remove duplicate resources across sources
4. **Validation**: Validate resource data before storage
5. **AWS Integration**: Push to SQS, store in S3, index in OpenSearch
6. **Monitoring**: CloudWatch metrics and alerts
7. **Scheduling**: EventBridge rules for periodic runs

## Dependencies

```
httpx>=0.24.0          # HTTP client for API requests
scrapy>=2.11.0         # Web scraping framework (RapidAPI only)
pydantic>=2.0.0        # Data validation
pydantic-settings>=2.0.0  # Settings management
```

## Configuration

Environment variables (optional):

```bash
# Optional tokens for higher rate limits
INGESTION_HUGGINGFACE_API_TOKEN=your_token_here
INGESTION_GITHUB_API_TOKEN=your_token_here

# AWS configuration (for production)
INGESTION_SQS_QUEUE_URL=https://sqs.region.amazonaws.com/account/queue
```

## Testing

```bash
# Test HTTP fetchers
python ingestion/test_http_fetchers.py

# Test individual components
python -m pytest ingestion/tests/

# Test with coverage
python -m pytest --cov=ingestion ingestion/tests/
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'httpx'"
```bash
cd backend
pip install -r requirements.txt
```

### "Rate limit exceeded"
Add authentication tokens to `.env` file

### "Connection timeout"
Check internet connection and API status

### "Scrapy not found"
Make sure you're in the `backend/ingestion` directory where `scrapy.cfg` is located
