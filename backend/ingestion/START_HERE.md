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

## Output Files

Files saved to `backend/ingestion/output/`:

| File | Content | Source |
|------|---------|--------|
| `models.json` | All models (deduplicated) | HuggingFace + OpenRouter |
| `huggingface_datasets.json` | Datasets | HuggingFace only |
| `kaggle_datasets.json` | Datasets | Kaggle only |
| `github_resources.json` | Repositories & tools | GitHub |

## Data Sources

| Source | What It Fetches | Auth Required |
|--------|-----------------|---------------|
| HuggingFace | Models & Datasets | ❌ No |
| OpenRouter | Models only | ❌ No |
| GitHub | Repositories | ⚠️ Optional |
| Kaggle | Datasets only | ✅ Yes |

## Deduplication

Models from HuggingFace and OpenRouter are automatically deduplicated:
- Uses model name as unique key
- If duplicate found, keeps the one with more downloads/stars
- Saves to single `models.json` file

## Authentication

| Source | Required? | How to Set Up |
|--------|-----------|---------------|
| HuggingFace | ❌ No | Public API, no auth needed |
| OpenRouter | ❌ No | Public API, no auth needed |
| GitHub | ⚠️ Optional | Add `INGESTION_GITHUB_API_TOKEN` to `.env` |
| Kaggle | ✅ Yes | Set up `~/.kaggle/kaggle.json` |

### GitHub Token (Optional)
Increases rate limit from 60 to 5000 requests/hour.

Get token: https://github.com/settings/tokens

Add to `backend/.env`:
```bash
INGESTION_GITHUB_API_TOKEN=your_token_here
```

### Kaggle Credentials (Required for Kaggle)
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Place in `~/.kaggle/kaggle.json`

## File Structure

```
backend/ingestion/
├── fetchers/
│   ├── huggingface_fetcher.py  # Models + Datasets
│   ├── openrouter_fetcher.py   # Models only
│   ├── github_fetcher.py       # Repositories
│   └── kaggle_fetcher.py       # Datasets only
│
├── output/
│   ├── models.json                    # Deduplicated models
│   ├── huggingface_datasets.json      # HF datasets
│   ├── kaggle_datasets.json           # Kaggle datasets
│   └── github_resources.json          # GitHub repos
│
├── run_ingestion.py            # Main runner
└── test_http_fetchers.py       # Test script
```

## Performance

- HuggingFace: ~30s for 1000+ models + 100 datasets
- OpenRouter: ~2s for 100+ models
- GitHub: ~2m for 900+ repos
- Kaggle: ~10s for 500 datasets

**Total: ~3-5 minutes**

## Troubleshooting

### Missing dependencies?
```bash
cd backend
pip install -r requirements.txt
pip install kaggle  # Optional, for Kaggle datasets
```

### GitHub rate limit?
Add token to `backend/.env`

### Kaggle not working?
Make sure credentials are set up at `~/.kaggle/kaggle.json`

---

**Ready?** Run `test_ingestion.bat` (Windows) or `python ingestion/test_http_fetchers.py` (Linux/Mac)
