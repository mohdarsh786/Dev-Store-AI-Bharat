# Ingestion Pipeline Setup Summary

## What We've Built

A comprehensive data ingestion pipeline that fetches developer resources from multiple external APIs and web sources.

## Data Sources

### 1. HuggingFace API (No Auth Required ✓)
- **What**: ML models and datasets
- **Method**: Direct API calls via Scrapy spider
- **Spider**: `huggingface_resource`
- **Auth**: None required (public API)
- **Fetches**: 
  - Pre-trained models (by task: text-classification, NLP, vision, etc.)
  - Public datasets
  - Model metadata (downloads, stars, license)

### 2. OpenRouter API (No Auth Required ✓)
- **What**: LLM models with pricing information
- **Method**: Direct API calls via Scrapy spider
- **Spider**: `openrouter`
- **Auth**: None required (public endpoint)
- **Fetches**:
  - Available LLM models (GPT, Claude, Llama, etc.)
  - Pricing per token
  - Context windows and capabilities

### 3. GitHub API (Optional Auth)
- **What**: Repositories, APIs, ML frameworks
- **Method**: Search API via Scrapy spider
- **Spider**: `github_resource`
- **Auth**: Optional token (increases rate limit from 60 to 5000 req/hour)
- **Fetches**:
  - Popular repositories by category
  - Stars, forks, and activity metrics
  - Repository metadata and topics

### 4. RapidAPI Marketplace (Optional Auth)
- **What**: Public APIs across various categories
- **Method**: Web scraping via Scrapy spider
- **Spider**: `rapidapi_resource`
- **Auth**: Optional API key
- **Fetches**:
  - API listings by category
  - Pricing and rating information
  - API metadata

## Architecture

```
External APIs
    ↓
Scrapy Spiders (4 spiders)
    ↓
Deduplication Pipeline (Redis)
    ↓
SQS Queue
    ↓
Batch Processor Worker
    ↓
Embedding Generation (Bedrock)
    ↓
Storage (PostgreSQL + OpenSearch)
```

## Files Created/Modified

### New Files
1. `backend/ingestion/scrapers/openrouter_spider.py` - OpenRouter API spider
2. `backend/ingestion/services/api_clients.py` - Direct API clients (alternative to Scrapy)
3. `backend/ingestion/run_spiders.py` - Script to run all spiders
4. `backend/ingestion/test_apis.py` - API connectivity test script
5. `backend/ingestion/SETUP_GUIDE.md` - Detailed setup instructions

### Modified Files
1. `backend/ingestion/config.py` - Added OpenRouter configuration
2. `backend/ingestion/scrapers/huggingface_spider.py` - Fixed imports and API calls
3. `backend/ingestion/scrapers/rapidapi_spider.py` - Enhanced scraping logic
4. `backend/ingestion/scrapers/github_spider.py` - Fixed imports
5. `backend/.env.example` - Added API key configuration
6. `backend/requirements.txt` - Added Scrapy dependency
7. `backend/ingestion/README.md` - Updated with API integration docs

## Configuration Required

Add these to your `.env` file (most are optional):

```bash
# GitHub API Token (optional - increases rate limit)
INGESTION_GITHUB_API_TOKEN=your_github_token

# HuggingFace API Token (optional - not required for public API)
# INGESTION_HUGGINGFACE_API_TOKEN=your_huggingface_token

# OpenRouter API Key (optional - not required for listing models)
# INGESTION_OPENROUTER_API_KEY=your_openrouter_key

# RapidAPI Key (optional)
# INGESTION_RAPIDAPI_KEY=your_rapidapi_key

# SQS Queue (required for production)
INGESTION_SQS_QUEUE_URL=your_sqs_queue_url
```

**Important**: HuggingFace and OpenRouter APIs work without authentication! You can start scraping immediately.

## How to Use

### 1. Test API Connectivity
```bash
python backend/ingestion/test_apis.py
```

### 2. Run Individual Spider
```bash
python -m scrapy crawl huggingface_resource -o output.json
```

### 3. Run All Spiders
```bash
python backend/ingestion/run_spiders.py
```

### 4. Use Direct API Clients (Alternative)
```python
from ingestion.services.api_clients import HuggingFaceAPIClient

client = HuggingFaceAPIClient()
models = await client.fetch_models(limit=100)
```

## Two Approaches Available

### Approach 1: Scrapy Spiders (Recommended for Production)
- Better for large-scale scraping
- Built-in rate limiting and retry logic
- Pipeline integration (deduplication, SQS)
- Handles pagination automatically

### Approach 2: Direct API Clients
- Simpler for one-off fetches
- Async/await support
- Good for testing and development
- Less overhead

## Next Steps

1. **Get API Keys**: Follow SETUP_GUIDE.md to obtain all required API keys
2. **Test Connectivity**: Run test_apis.py to verify all APIs are accessible
3. **Run Spiders**: Test individual spiders with sample data
4. **Set Up AWS**: Configure SQS, Redis, and OpenSearch
5. **Deploy Workers**: Set up batch processing and embedding generation
6. **Schedule**: Configure EventBridge for automated runs

## Resource Categories

The pipeline categorizes resources into:
- **model**: ML models (HuggingFace, OpenRouter)
- **dataset**: Training datasets (HuggingFace)
- **api**: Public APIs (RapidAPI, GitHub)
- **solution**: Complete solutions and frameworks (GitHub)

## Data Flow

1. **Scraping**: Spiders fetch raw data from APIs
2. **Normalization**: Convert to standard resource format
3. **Deduplication**: Check Redis for duplicates (SHA256 hash)
4. **Queueing**: Send to SQS for processing
5. **Batch Processing**: Worker processes batches of 10-20
6. **Embedding**: Generate semantic embeddings via Bedrock
7. **Storage**: Save to PostgreSQL with metadata
8. **Indexing**: Index in OpenSearch for search
9. **Caching**: Cache popular queries in Redis

## Monitoring

- CloudWatch logs: `/aws/ecs/devstore-scraper`
- Metrics: scrape_duration, records_fetched, errors
- SQS queue depth monitoring
- Redis deduplication hit rate

## Support

See `backend/ingestion/SETUP_GUIDE.md` for detailed setup instructions and troubleshooting.
