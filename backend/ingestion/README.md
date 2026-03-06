# DevStore Ingestion Pipeline

## Quick Start

Test the ingestion pipeline locally:

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Test HTTP fetchers
.venv\Scripts\python.exe ingestion\test_http_fetchers.py

# 3. Run full ingestion
.venv\Scripts\python.exe ingestion\run_ingestion.py
```

## Data Sources

| Source | Method | Auth Required | Output File |
|--------|--------|---------------|-------------|
| HuggingFace Models | HTTP API | ❌ No | `data/models.json` |
| OpenRouter Models | HTTP API | ❌ No | `data/models.json` |
| HuggingFace Datasets | HTTP API | ❌ No | `data/huggingface_datasets.json` |
| Kaggle Datasets | Kaggle API | ⚠️ Yes | `data/kaggle_datasets.json` |
| GitHub Repos/Tools | HTTP API | ⚠️ Optional | `data/github_repos.json` |

## Output Structure

All data is saved to `backend/ingestion/data/`:

```
data/
├── models.json                    # Models from HuggingFace + OpenRouter (deduplicated)
├── huggingface_datasets.json     # Datasets from HuggingFace
├── kaggle_datasets.json          # Datasets from Kaggle
└── github_repos.json             # Repositories and tools from GitHub
```

## Authentication

### HuggingFace (No Auth Required)
- Public API, no authentication needed
- Works out of the box

### OpenRouter (No Auth Required)
- Public models endpoint
- No API key needed

### GitHub (Optional Token)
- Works without token: 60 requests/hour
- With token: 5000 requests/hour
- Get token: https://github.com/settings/tokens

Add to `backend/.env`:
```bash
INGESTION_GITHUB_API_TOKEN=your_token_here
```

### Kaggle (API Credentials Required)
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Place in:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`

## File Structure

```
backend/ingestion/
├── fetchers/                    # HTTP-based fetchers
│   ├── huggingface_fetcher.py  # HuggingFace API
│   ├── openrouter_fetcher.py   # OpenRouter API
│   ├── github_fetcher.py       # GitHub API
│   └── kaggle_fetcher.py       # Kaggle API
│
├── services/                    # Shared services
│   └── storage_service.py      # JSON file storage & deduplication
│
├── data/                        # Output directory (gitignored)
│   ├── models.json
│   ├── huggingface_datasets.json
│   ├── kaggle_datasets.json
│   └── github_repos.json
│
├── run_ingestion.py            # Main runner
├── test_http_fetchers.py       # Test script
└── config.py                   # Configuration
```

## Features

### Deduplication
Models from HuggingFace and OpenRouter are automatically deduplicated based on `source_url` before saving to `models.json`.

### Error Handling
- Graceful error handling for each source
- Pipeline continues even if one source fails
- Detailed error messages and logging

### Data Normalization
All resources are normalized to a standard format:

```json
{
  "name": "resource-name",
  "description": "Description",
  "source": "huggingface|openrouter|github|kaggle",
  "source_url": "https://...",
  "author": "creator",
  "stars": 1234,
  "downloads": 5678,
  "license": "MIT",
  "tags": ["tag1", "tag2"],
  "category": "model|dataset|solution",
  "metadata": {...}
}
```

## Performance

- **HuggingFace**: ~30s for 1000+ models + 100 datasets
- **OpenRouter**: ~2s for 100+ models
- **GitHub**: ~2m for 900+ repositories
- **Kaggle**: ~1m for 500+ datasets

**Total: ~3-5 minutes**

## Troubleshooting

### "ModuleNotFoundError: No module named 'httpx'"
```bash
cd backend
pip install -r requirements.txt
```

### "ModuleNotFoundError: No module named 'kaggle'"
```bash
pip install kaggle
```

### "Kaggle authentication failed"
Set up Kaggle API credentials:
1. Download `kaggle.json` from https://www.kaggle.com/settings/account
2. Place in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

### "GitHub rate limit exceeded"
Add GitHub token to `backend/.env`:
```bash
INGESTION_GITHUB_API_TOKEN=your_token_here
```

## Next Steps

1. Run the test script to verify everything works
2. Run full ingestion to fetch all data
3. Review output JSON files in `data/` directory
4. Integrate with your backend database
5. Set up AWS infrastructure for production

## Documentation

- **[START_HERE.md](START_HERE.md)** - Quick start guide
- **[QUICKSTART.md](QUICKSTART.md)** - Detailed guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Architecture details
- **[CLEAN_STRUCTURE.md](CLEAN_STRUCTURE.md)** - Codebase structure
