# Ingestion Pipeline Architecture

## Overview

The Dev Store ingestion pipeline fetches ML models, datasets, and tools from multiple external sources using direct HTTP API calls. Data is automatically deduplicated and organized into separate JSON files.

## Design Decision: HTTP API Calls

We use direct HTTP requests (`httpx` library) instead of web scraping frameworks because:

1. **Simplicity**: Direct API calls are simpler and more maintainable
2. **Performance**: Faster for REST APIs (no framework overhead)
3. **Reliability**: Official APIs are more stable than HTML scraping
4. **Error Handling**: Better control over retries and rate limiting
5. **Dependencies**: Lightweight (just `httpx` and `kaggle`)

## Data Sources

| Source | Method | Auth Required | What It Fetches |
|--------|--------|---------------|-----------------|
| HuggingFace | HTTP API | ❌ No | Models + Datasets |
| OpenRouter | HTTP API | ❌ No | Models only |
| GitHub | HTTP API | ⚠️ Optional | Repositories |
| Kaggle | Kaggle API | ✅ Yes | Datasets only |

## Output Structure

### 4 JSON Files

```
backend/ingestion/output/
├── models.json                    # Deduplicated models (HF + OR)
├── huggingface_datasets.json      # HuggingFace datasets
├── kaggle_datasets.json           # Kaggle datasets
└── github_resources.json          # GitHub repositories
```

### File Contents

1. **models.json**
   - Source: HuggingFace + OpenRouter
   - Content: All ML/LLM models
   - Deduplication: By model name (keeps higher downloads+stars)

2. **huggingface_datasets.json**
   - Source: HuggingFace only
   - Content: ML datasets

3. **kaggle_datasets.json**
   - Source: Kaggle only
   - Content: Data science datasets

4. **github_resources.json**
   - Source: GitHub only
   - Content: Repositories and tools

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ├─────────────────────────────────┐
                              │                                 │
                    ┌─────────▼─────────┐          ┌───────────▼──────────┐
                    │  HTTP FETCHERS    │          │  KAGGLE API CLIENT   │
                    │  (httpx)          │          │  (kaggle package)    │
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
└────────┬───────┘  └─────────┬───────┘  └─────────┬───────┘ │
         │                    │                     │          │
         │                    │                     │   ┌──────▼──────────┐
         │                    │                     │   │  Kaggle Fetcher │
         │                    │                     │   │                 │
         │                    │                     │   │ • Datasets      │
         │                    │                     │   └──────┬──────────┘
         │                    │                     │          │
         ├────────────────────┤                     │          │
         │                    │                     │          │
         ▼                    ▼                     ▼          ▼
    ┌────────────┐      ┌──────────┐         ┌─────────┐  ┌─────────┐
    │  Models    │      │  Models  │         │ GitHub  │  │ Kaggle  │
    │ (HF API)   │      │ (OR API) │         │  Repos  │  │Datasets │
    └─────┬──────┘      └─────┬────┘         └────┬────┘  └────┬────┘
          │                   │                   │            │
          └───────┬───────────┘                   │            │
                  │                               │            │
                  ▼                               │            │
         ┌────────────────┐                       │            │
         │ Deduplication  │                       │            │
         │  (by name)     │                       │            │
         └────────┬───────┘                       │            │
                  │                               │            │
                  ▼                               ▼            ▼
         ┌────────────────┐              ┌────────────┐  ┌─────────┐
         │  models.json   │              │github_     │  │kaggle_  │
         │  (deduplicated)│              │resources   │  │datasets │
         └────────────────┘              │.json       │  │.json    │
                                         └────────────┘  └─────────┘
         ┌────────────────┐
         │ HF Datasets    │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │huggingface_    │
         │datasets.json   │
         └────────────────┘
```

## Components

### 1. HTTP Fetchers (`fetchers/`)

Direct API clients using `httpx` library:

#### HuggingFace Fetcher
- **File**: `fetchers/huggingface_fetcher.py`
- **API**: `https://huggingface.co/api`
- **Auth**: Not required (public API)
- **Fetches**:
  - Models by task (text-classification, image-classification, etc.)
  - Datasets sorted by downloads
- **Output**: Models → `models.json`, Datasets → `huggingface_datasets.json`

#### OpenRouter Fetcher
- **File**: `fetchers/openrouter_fetcher.py`
- **API**: `https://openrouter.ai/api/v1`
- **Auth**: Not required (public endpoint)
- **Fetches**:
  - All available LLM models
  - Pricing information
  - Context windows
- **Output**: Models → `models.json` (deduplicated with HuggingFace)

#### GitHub Fetcher
- **File**: `fetchers/github_fetcher.py`
- **API**: `https://api.github.com`
- **Auth**: Optional (60→5000 req/hour with token)
- **Fetches**:
  - Repositories by search queries
  - Stars, forks, metadata
- **Output**: Repositories → `github_resources.json`

#### Kaggle Fetcher
- **File**: `fetchers/kaggle_fetcher.py`
- **API**: Kaggle API (via `kaggle` package)
- **Auth**: Required (API credentials)
- **Fetches**:
  - Datasets sorted by popularity
  - Metadata and statistics
- **Output**: Datasets → `kaggle_datasets.json`

### 2. Deduplication Logic

Models from HuggingFace and OpenRouter are automatically deduplicated:

```python
def deduplicate_models(hf_models, or_models):
    """
    Deduplication strategy:
    1. Use model name (lowercase) as unique key
    2. If duplicate found, compare popularity score
    3. Keep model with higher (downloads + stars)
    """
    models_dict = {}
    
    for model in hf_models + or_models:
        name = model['name'].lower()
        
        if name not in models_dict:
            models_dict[name] = model
        else:
            existing_score = existing['downloads'] + existing['stars']
            new_score = model['downloads'] + model['stars']
            
            if new_score > existing_score:
                models_dict[name] = model
    
    return list(models_dict.values())
```

### 3. Data Normalization

All fetchers normalize data to a standard format:

```python
{
    'name': str,              # Resource name
    'description': str,       # Description
    'source': str,           # huggingface|openrouter|github|kaggle
    'source_url': str,       # Original URL
    'author': str,           # Creator/organization
    'stars': int,            # Popularity metric
    'downloads': int,        # Download count
    'license': str,          # License type
    'tags': List[str],       # Tags/topics (max 10)
    'version': str,          # Version identifier
    'category': str,         # model|dataset|api|solution
    'thumbnail_url': str,    # Image URL
    'readme_url': str,       # Documentation URL
    'metadata': dict,        # Source-specific metadata
    'scraped_at': str        # ISO timestamp
}
```

## File Structure

```
backend/ingestion/
├── fetchers/                       # HTTP-based fetchers
│   ├── __init__.py
│   ├── huggingface_fetcher.py     # HuggingFace API
│   ├── openrouter_fetcher.py      # OpenRouter API
│   ├── github_fetcher.py          # GitHub API
│   └── kaggle_fetcher.py          # Kaggle API
│
├── output/                         # Output directory (created at runtime)
│   ├── models.json                # Deduplicated models
│   ├── huggingface_datasets.json  # HuggingFace datasets
│   ├── kaggle_datasets.json       # Kaggle datasets
│   └── github_resources.json      # GitHub repositories
│
├── run_ingestion.py               # Main runner script
├── test_http_fetchers.py          # Test HTTP fetchers
├── test_imports.py                # Test imports
│
├── START_HERE.md                  # Quick start guide
├── QUICKSTART.md                  # Detailed guide
├── ARCHITECTURE.md                # This file
├── DATA_STRUCTURE.md              # Data structure docs
└── README.md                      # Full documentation
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
# results['models'] → models.json
# results['datasets'] → huggingface_datasets.json

# OpenRouter
from fetchers.openrouter_fetcher import OpenRouterFetcher
fetcher = OpenRouterFetcher()
models = fetcher.fetch_and_normalize_all()
# models → models.json (deduplicated)

# GitHub
from fetchers.github_fetcher import GitHubFetcher
fetcher = GitHubFetcher()
repos = fetcher.fetch_and_normalize_all()
# repos → github_resources.json

# Kaggle
from fetchers.kaggle_fetcher import KaggleFetcher
fetcher = KaggleFetcher()
datasets = fetcher.fetch_and_normalize_all()
# datasets → kaggle_datasets.json
```

## Performance

| Source | Method | Time | Resources |
|--------|--------|------|-----------|
| HuggingFace | HTTP | ~30s | ~1000 models + 100 datasets |
| OpenRouter | HTTP | ~2s | ~100 models |
| GitHub | HTTP | ~2m | ~900 repositories |
| Kaggle | Kaggle API | ~10s | ~500 datasets |

**Total**: ~3-5 minutes for all sources

**Deduplication**: Adds ~1 second (in-memory operation)

## Error Handling

### HTTP Fetchers
- Automatic retries with exponential backoff
- Rate limit detection and handling
- Graceful degradation (continue on error)
- Detailed error logging

### Deduplication
- Handles missing fields gracefully
- Defaults to 0 for missing downloads/stars
- Preserves all unique models

## Rate Limits

| Source | Without Token | With Token | How to Get Token |
|--------|---------------|------------|------------------|
| HuggingFace | Unlimited | N/A | Not required |
| OpenRouter | Unlimited | N/A | Not required |
| GitHub | 60/hour | 5000/hour | https://github.com/settings/tokens |
| Kaggle | N/A | Required | https://www.kaggle.com/docs/api |

## Authentication Setup

### GitHub (Optional)
```bash
# Add to backend/.env
INGESTION_GITHUB_API_TOKEN=your_token_here
```

### Kaggle (Required)
```bash
# Option 1: Place kaggle.json in ~/.kaggle/
~/.kaggle/kaggle.json

# Option 2: Set environment variables
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

## Dependencies

```
httpx>=0.24.0          # HTTP client for API requests
kaggle>=1.5.0          # Kaggle API client
pydantic>=2.0.0        # Data validation (optional)
```

## Future Enhancements

1. **Incremental Updates**: Only fetch new/updated resources
2. **Caching**: Cache API responses to reduce API calls
3. **Advanced Deduplication**: Use fuzzy matching for similar names
4. **Validation**: Validate resource data before storage
5. **AWS Integration**: Push to SQS, store in S3, index in OpenSearch
6. **Monitoring**: CloudWatch metrics and alerts
7. **Scheduling**: EventBridge rules for periodic runs
8. **Parallel Fetching**: Fetch from multiple sources simultaneously

## Benefits

1. **No Duplicates**: Automatic deduplication for models
2. **Clean Separation**: Datasets from different sources in separate files
3. **Simple**: Direct HTTP calls, no complex frameworks
4. **Fast**: Optimized for REST APIs
5. **Reliable**: Uses official APIs
6. **Maintainable**: Clear code structure
7. **Flexible**: Easy to add new sources
8. **Traceable**: Each item has `source` field

## Testing

```bash
# Test all fetchers
python ingestion/test_http_fetchers.py

# Test imports
python ingestion/test_imports.py

# Run full ingestion
python ingestion/run_ingestion.py
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'httpx'"
```bash
cd backend
pip install -r requirements.txt
```

### "Rate limit exceeded" (GitHub)
Add authentication token to `.env` file

### "Kaggle authentication failed"
Set up Kaggle API credentials at `~/.kaggle/kaggle.json`

### "Connection timeout"
Check internet connection and API status

---

**The architecture is optimized for simplicity, performance, and reliability!**
