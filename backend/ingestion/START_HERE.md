# 🚀 Start Here

## Quick Test (30 seconds)

```bash
# Windows
cd backend
test_ingestion.bat

# Linux/Mac
cd backend
source .venv/bin/activate
python ingestion/test_http_fetchers.py
```

## Run Full Ingestion (3-5 minutes)

```bash
# Windows
cd backend
run_full_ingestion.bat

# Linux/Mac
cd backend
source .venv/bin/activate
python ingestion/run_ingestion.py
```

## What This Does

### HTTP Fetchers (Fast & Simple)
- **HuggingFace**: Fetches ML models & datasets via API (NO AUTH REQUIRED)
- **OpenRouter**: Fetches LLM models with pricing via API (NO AUTH REQUIRED)
- **GitHub**: Fetches repositories & tools via API (optional token)

### Scrapy Crawler (Web Scraping)
- **RapidAPI**: Scrapes API marketplace (no official API available)

## Authentication

| Source | Required? | Purpose |
|--------|-----------|---------|
| HuggingFace | ❌ No | Public API, no auth needed |
| OpenRouter | ❌ No | Public API, no auth needed |
| GitHub | ⚠️ Optional | Increases rate limit 60→5000/hour |
| RapidAPI | ❌ No | Web scraping, no API |

## Output

Files saved to `backend/ingestion/output/`:
- `huggingface_resources.json` - Models & datasets
- `openrouter_resources.json` - LLM models
- `github_resources.json` - Repositories
- `rapidapi_resources.json` - APIs
- `all_resources_combined.json` - Everything

## File Structure

```
backend/ingestion/
├── fetchers/                    # HTTP-based fetchers
│   ├── huggingface_fetcher.py  # HuggingFace API
│   ├── openrouter_fetcher.py   # OpenRouter API
│   └── github_fetcher.py       # GitHub API
│
├── scrapers/                    # Scrapy spiders
│   └── rapidapi_spider.py      # RapidAPI scraper
│
├── run_ingestion.py            # Main runner
├── test_http_fetchers.py       # Test script
└── output/                     # Output directory
```

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Architecture details
- **[README.md](README.md)** - Full documentation

## Troubleshooting

### Missing dependencies?
```bash
cd backend
pip install -r requirements.txt
```

### GitHub rate limit?
Add token to `backend/.env`:
```bash
INGESTION_GITHUB_API_TOKEN=your_token_here
```

Get token: https://github.com/settings/tokens

## Performance

- HuggingFace: ~30s for 1000+ resources
- OpenRouter: ~2s for 100+ models
- GitHub: ~2m for 900+ repos
- RapidAPI: Varies (web scraping)

**Total: ~3-5 minutes**

---

**Ready?** Run `test_ingestion.bat` (Windows) or `python ingestion/test_http_fetchers.py` (Linux/Mac)
