# Quick Start Guide

Get started with the ingestion pipeline in 2 minutes!

## Architecture

- **HuggingFace, OpenRouter, GitHub**: Direct HTTP API requests (fast, simple)
- **RapidAPI**: Scrapy web crawler (no official API available)

## Prerequisites

```bash
cd backend
pip install -r requirements.txt
```

## Option 1: Test HTTP Fetchers (Recommended)

Test the HTTP-based fetchers:

```bash
# From backend directory
.venv\Scripts\python.exe ingestion\test_http_fetchers.py

# Or on Linux/Mac
source .venv/bin/activate
python ingestion/test_http_fetchers.py
```

This will:
- ✅ Test HuggingFace API (NO AUTH REQUIRED)
- ✅ Test OpenRouter API (NO AUTH REQUIRED)
- ✅ Test GitHub API (optional token)
- ✅ Fetch sample data from each API
- ✅ Verify data normalization

## Option 2: Run Full Ingestion

Run the complete ingestion pipeline:

```bash
# From backend directory
.venv\Scripts\python.exe ingestion\run_ingestion.py

# Or on Linux/Mac
source .venv/bin/activate
python ingestion/run_ingestion.py
```

This will:
1. Fetch models and datasets from HuggingFace (HTTP)
2. Fetch LLM models from OpenRouter (HTTP)
3. Fetch repositories from GitHub (HTTP)
4. Scrape APIs from RapidAPI (Scrapy crawler)
5. Save all data to JSON files in `output/` directory

Output files:
- `huggingface_resources.json` - Models and datasets
- `openrouter_resources.json` - LLM models with pricing
- `github_resources.json` - Repositories and tools
- `rapidapi_resources.json` - API marketplace listings
- `all_resources_combined.json` - All resources combined

## Option 3: Run Only RapidAPI Crawler

Run just the Scrapy crawler for RapidAPI:

```bash
cd backend/ingestion
scrapy crawl rapidapi_resource -o output/rapidapi.json
```

## Authentication

| Source | Required? | How to Get | Purpose |
|--------|-----------|------------|---------|
| HuggingFace | ❌ No | N/A | Public API |
| OpenRouter | ❌ No | N/A | Public API |
| GitHub | ⚠️ Optional | [Get token](https://github.com/settings/tokens) | 60→5000 req/hour |
| RapidAPI | ❌ No | N/A | Web scraping |

To add GitHub token (optional):

```bash
# Add to backend/.env
INGESTION_GITHUB_API_TOKEN=your_token_here
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'httpx'"

Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

### "Rate limit exceeded" (GitHub only)

Add a GitHub token to `.env` to increase rate limit from 60 to 5000 requests/hour.

### "Scrapy 2.11.0 - no active project" (RapidAPI)

Make sure you're in the `backend/ingestion` directory where `scrapy.cfg` is located.

## Next Steps

1. Review output JSON files to see the data structure
2. Integrate with your database/storage layer
3. Set up AWS infrastructure for production (SQS, OpenSearch, etc.)
4. Schedule regular runs with cron or EventBridge

## Quick Reference

| Command | Purpose | Method |
|---------|---------|--------|
| `python ingestion/test_http_fetchers.py` | Test HTTP fetchers | HTTP |
| `python ingestion/run_ingestion.py` | Full pipeline | HTTP + Scrapy |
| `scrapy crawl rapidapi_resource` | RapidAPI only | Scrapy |

## Sample Output

```json
{
  "name": "bert-base-uncased",
  "description": "BERT base model (uncased)",
  "source": "huggingface",
  "source_url": "https://huggingface.co/bert-base-uncased",
  "author": "google",
  "stars": 1234,
  "downloads": 5678900,
  "license": "apache-2.0",
  "tags": ["text-classification", "pytorch", "transformers"],
  "category": "model",
  "metadata": {
    "model_id": "bert-base-uncased",
    "pipeline_tag": "fill-mask",
    "library_name": "transformers"
  }
}
```

## Performance

- **HuggingFace**: ~1000 models + 100 datasets in ~30 seconds
- **OpenRouter**: ~100 models in ~2 seconds
- **GitHub**: ~900 repositories in ~2 minutes
- **RapidAPI**: Varies (web scraping, depends on site speed)

Total time: ~3-5 minutes for all sources
