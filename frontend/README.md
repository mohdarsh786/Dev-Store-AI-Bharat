# DevStore — AI for Bharat 🇮🇳
### Next.js 14 · App Router · AWS Bedrock · OpenSearch · TypeScript

> **The premier AI developer marketplace for Indian builders.** Discover, rank, and integrate APIs, ML Models, and Datasets — all curated for the Bharat ecosystem. Backed by a RAG pipeline powered by AWS Bedrock (Claude 3 Sonnet + Titan Embeddings) and OpenSearch semantic vector search.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Unified Ranking System](#2-unified-ranking-system)
3. [Security Protocols](#3-security-protocols)
4. [Quickstart Guide](#4-quickstart-guide)
5. [API Endpoints](#5-api-endpoints)
6. [Data Ingestion Pipeline](#6-data-ingestion-pipeline)
7. [Backend Reference](#7-backend-reference)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Architecture Overview

### System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Browser (Client Layer)                            │
│  DevStoreDashboard.jsx — SWR hooks → fetch("/api/*")                │
│  Hinglish ghost-text · Bento Grid · Glassmorphism · Theme toggle     │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ HTTPS — same-origin /api/* calls
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Next.js 14 App Router (Server Layer)                    │
│                                                                      │
│  app/api/search/route.ts     ← POST /api/search                      │
│  app/api/trending/route.ts   ← GET  /api/trending?sort=rank_score    │
│  app/api/resources/route.ts  ← GET  /api/resources                   │
│  app/api/health/route.ts     ← GET  /api/health                      │
│                                                                      │
│  ⚑ AWS credentials NEVER leave this layer                           │
│  ⚑ BEDROCK_MODEL_ID NEVER bundled to client                         │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ Internal HTTP — localhost:8000
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                FastAPI Backend (Python)                              │
│                                                                      │
│  POST /api/v1/search   → Intent Extraction (Claude 3)               │
│                        → Embedding (Titan Text v1)                   │
│                        → KNN Vector Search (OpenSearch)              │
│                        → Composite Ranking (rank_score)              │
│                                                                      │
│  GET  /api/v1/trending → Queries rank_score, trending_score,         │
│                          category_rank from PostgreSQL               │
└──────────────────────────┬──────────────┬────────────────────────────┘
                           │              │
              ┌────────────▼──┐    ┌──────▼───────────────┐
              │  PostgreSQL   │    │  AWS OpenSearch       │
              │  (RDS Aurora) │    │  (Vector + BM25)      │
              │  rank_score   │    │  devstore_resources   │
              │  trending_sc. │    │  KNN index            │
              │  category_rank│    └──────────────────────┘
              └───────────────┘
```

### The Handshake — How Credentials Are Protected

The Next.js Route Handlers act as a **secure server-side proxy**. When the browser calls `fetch("/api/search")`, the Route Handler at `app/api/search/route.ts` reads `process.env.BACKEND_URL` (server-only) and forwards the request to FastAPI. The browser never sees `BEDROCK_MODEL_ID`, `AWS_ACCESS_KEY_ID`, `OPENSEARCH_HOST`, or any other secret.

```
Browser          Next.js Server         FastAPI + AWS
   │                    │                     │
   │  POST /api/search  │                     │
   │───────────────────▶│                     │
   │                    │  POST :8000/v1/srch │
   │                    │────────────────────▶│
   │                    │                     │  InvokeModel (Bedrock)
   │                    │                     │  KNNSearch (OpenSearch)
   │                    │◀────────────────────│
   │◀───────────────────│                     │
```

---

## 2. Unified Ranking System

### Overview

Every resource exposes three computed fields that power the discovery pills:

| Field | Type | Discovery Pill | Update Cadence |
|-------|------|----------------|----------------|
| `rank_score` | `FLOAT 0–1` | **Trending** (default sort) | Daily cron |
| `trending_score` | `FLOAT 0–1` | **Trending** filter | Daily cron |
| `category_rank` | `INTEGER` | `#N IN APIS` badge | Daily cron |
| `boilerplate_download_count` | `INTEGER` | **Most Popular** sort | Real-time |

### Composite `rank_score` Formula

```
rank_score =
  semantic_relevance  × 0.40   ← cosine similarity from OpenSearch KNN
  popularity_score    × 0.30   ← log(github_stars + downloads)
  optimization_score  × 0.20   ← latency, uptime, SLA tier
  freshness_score     × 0.10   ← recency decay function
```

### `trending_score` Formula

```
trending_score =
  log_scale(recent_downloads_7d) × 0.40
  log_scale(recent_views_7d)     × 0.30
  log_scale(recent_bookmarks)    × 0.20
  sigmoid(growth_rate)           × 0.10
```
> Normalized to [0, 1]. Scores > 0.7 render the badge. Scores > 0.4 render .

### How Discovery Pills Query the Database

| UI Filter | FastAPI Query | SQL Logic |
|-----------|--------------|-----------|
| **Trending** | `GET /api/v1/trending?sort=rank_score` | `ORDER BY rank_score DESC` |
| **Top Free** | `GET /api/v1/trending?pricing_type=free&sort=popularity` | `WHERE pricing_type='free' ORDER BY rank_score DESC` |
| **Top Paid** | `GET /api/v1/trending?pricing_type=paid&sort=popularity` | `WHERE pricing_type='paid' ORDER BY rank_score DESC` |
| **Most Popular** | `GET /api/v1/trending?sort=downloads` | `ORDER BY boilerplate_download_count DESC` |

### `#Rank` Badge Logic (Frontend)

```js
// DevStoreDashboard.jsx — mapResource()
rank: r.rank || (index < 10 ? index + 1 : 0)

// ToolCard — renders badge only for rank 1–10
{tool.rank > 0 && tool.rank <= 10 && (
  <div>#{tool.rank} IN {tool.category}S</div>
)}
```

### Database Schema

```sql
-- Core ranking columns (applied via migration 008)
ALTER TABLE resources ADD COLUMN IF NOT EXISTS rank_score              FLOAT   DEFAULT 0.0;
ALTER TABLE resources ADD COLUMN IF NOT EXISTS trending_score          FLOAT   DEFAULT 0.0;
ALTER TABLE resources ADD COLUMN IF NOT EXISTS category_rank           INTEGER DEFAULT NULL;
ALTER TABLE resources ADD COLUMN IF NOT EXISTS boilerplate_download_count INTEGER DEFAULT 0;

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_resources_rank        ON resources(rank_score DESC);
CREATE INDEX IF NOT EXISTS idx_resources_trending    ON resources(trending_score DESC);
CREATE INDEX IF NOT EXISTS idx_resources_cat_rank    ON resources(type, category_rank);
CREATE INDEX IF NOT EXISTS idx_resources_downloads   ON resources(boilerplate_download_count DESC);
```

### Automating Rank Updates

```bash
# Run manually
cd backend && python update_rankings.py --time-window 7

# Cron (Linux/macOS) — daily at 2 AM
0 2 * * * cd /path/to/backend && python update_rankings.py >> /var/log/rankings.log 2>&1

# AWS Lambda handler
def lambda_handler(event, context):
    import asyncio
    from update_rankings import main
    asyncio.run(main())
    return {'statusCode': 200, 'body': 'Rankings updated'}
```

---

## 3. Security Protocols

### Mandatory `.env.local` Variables

> **These variables MUST be set in `nextjs-frontend/.env.local` before running the app.**  
> Never prefix AWS/OpenSearch vars with `NEXT_PUBLIC_` — that would expose them to the browser bundle.

```bash
# ── Core Backend Proxy ────────────────────────────────────────────────
BACKEND_URL=http://localhost:8000            # FastAPI base URL

# ── AWS Identity ─────────────────────────────────────────────────────
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=<your-access-key>         
AWS_SECRET_ACCESS_KEY=<your-secret-key>

# ── AWS Bedrock (server-side only) ───────────────────────────────────
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1

# ── OpenSearch ───────────────────────────────────────────────────────
OPENSEARCH_HOST=<your-domain>.us-east-1.es.amazonaws.com
OPENSEARCH_PORT=443
OPENSEARCH_USE_SSL=true
OPENSEARCH_INDEX_NAME=devstore_resources

# ── S3 Boilerplate ───────────────────────────────────────────────────
S3_BUCKET_BOILERPLATE=devstore-boilerplate-templates
S3_BUCKET_CRAWLER_DATA=devstore-crawler-data

# ── Database ─────────────────────────────────────────────────────────
DATABASE_URL=postgresql://user:password@<rds-host>:5432/devstore
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# ── Public vars (NEXT_PUBLIC_ prefix = safe to expose) ───────────────
NEXT_PUBLIC_APP_NAME=DevStore
NEXT_PUBLIC_APP_ENV=development
```

### IAM Minimum Permissions for Bedrock + OpenSearch

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockInference",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel"],
      "Resource": [
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
        "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1"
      ]
    },
    {
      "Sid": "OpenSearchSearch",
      "Effect": "Allow",
      "Action": ["es:ESHttpGet", "es:ESHttpPost"],
      "Resource": "arn:aws:es:us-east-1:<ACCOUNT>:domain/devstore-search/*"
    },
    {
      "Sid": "S3Boilerplate",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::devstore-boilerplate-templates/*"
    }
  ]
}
```

### Security Rules Summary

| Rule | Implementation |
|------|---------------|
| No secrets in browser | `NEXT_PUBLIC_` prefix only for safe vars |
| No secrets in Git | `.env.local` is `.gitignore`d |
| Server-side proxy | All AWS calls via `app/api/*` Route Handlers |
| Input sanitization | Pydantic validators on FastAPI side |
| Parameterized queries | `db.execute(query, (param,))` — no f-strings |
| Rate limiting | 100 req/min per IP (Redis-backed in FastAPI) |
| Structured logging | Never log API keys, PII, or credentials |

---

## 4. Quickstart Guide

### Prerequisites

- Node.js ≥ 18.17
- Python ≥ 3.11 (for backend)
- AWS account with Bedrock access enabled
- PostgreSQL instance (local or RDS)
- OpenSearch domain (local or AWS)

---

### Step 1 — Configure Environment

```bash
# Copy this block into nextjs-frontend/.env.local
# Then fill in your real values

BACKEND_URL=http://localhost:8000
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USE_SSL=false
OPENSEARCH_INDEX_NAME=devstore_resources
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/devstore
NEXT_PUBLIC_APP_NAME=DevStore
NEXT_PUBLIC_APP_ENV=development
```

---

### Step 2 — Start the Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
python run_migrations.py

# Seed initial rankings (first run only)
python update_rankings.py --time-window 7

# Start the FastAPI server
uvicorn main:app --reload --port 8000
```

> **Verify:** `http://localhost:8000/api/v1/health` should return `{"status":"healthy"}`

---

### Step 3 — Start the Frontend

```bash
cd nextjs-frontend

# Install dependencies (if not already done)
npm install

# Launch dev server (or run setup-frontend.bat on Windows)
npm run dev
```

> **Live at:** `http://localhost:3000`  
> **API routes live at:** `http://localhost:3000/api/*`

---

## 5. API Endpoints

### Next.js Route Handlers (`/api/*`)

These are the **only endpoints the browser calls**. They proxy to FastAPI server-side.

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/api/search` | Semantic search via Bedrock + OpenSearch |
| `GET` | `/api/trending` | Ranked resource list (supports `sort`, `pricing_type`, `resource_type`) |
| `GET` | `/api/resources` | Paginated resource list |
| `GET` | `/api/health` | Backend connectivity check |

### FastAPI Endpoints (`localhost:8000`)

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/api/v1/search` | Intent extraction → embedding → KNN → ranking |
| `GET` | `/api/v1/resources` | List with filters: `type`, `pricing`, `page` |
| `GET` | `/api/v1/resources/{id}` | Single resource detail |
| `GET` | `/api/v1/trending` | Top ranked, supports `sort=rank_score\|popularity\|downloads` |
| `POST` | `/api/v1/boilerplate/generate` | Generate code + track `boilerplate_download_count` |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/health/detailed` | Deep check: DB + Redis + OpenSearch |

### Search Request Schema

```bash
POST /api/search
Content-Type: application/json

{
  "query": "Bhai, best payment gateway batao...",
  "pricing_filter": "free",          # optional: free | paid | freemium
  "resource_types": ["API", "Model"], # optional array
  "limit": 20
}
```

### Trending Request Schema

```bash
GET /api/trending?resource_type=API&pricing_type=free&sort=rank_score&limit=40
```

---

## 6. Data Ingestion Pipeline

The backend populates OpenSearch and PostgreSQL via Scrapy spiders and a batch processor:

```
External Sources → Scrapy Spiders → Dedup (Redis) → SQS Queue
                                                         │
                                              Batch Processor Worker
                                                         │
                                         Bedrock (Titan Embeddings)
                                                         │
                                    PostgreSQL + OpenSearch Index
```

### Data Sources

| Source | Spider | Auth | Fetches |
|--------|--------|------|---------|
| HuggingFace | `huggingface_resource` | None (public) | Models, Datasets |
| OpenRouter | `openrouter` | None (public) | LLM pricing + specs |
| GitHub | `github_resource` | Optional token | Stars, forks, topics |
| RapidAPI | `rapidapi_resource` | Optional key | API listings, pricing |

### Running the Pipeline

```bash
# Test API connectivity
python backend/ingestion/test_apis.py

# Run a single spider
python -m scrapy crawl huggingface_resource -o output.json

# Run all spiders
python backend/ingestion/run_spiders.py

# Or use direct API clients
from ingestion.services.api_clients import HuggingFaceAPIClient
client = HuggingFaceAPIClient()
models = await client.fetch_models(limit=100)
```

---

## 7. Backend Reference

### Common Commands

```bash
# Start backend with auto-reload
cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
cd backend && pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_ranking_features.py

# Manual ranking update
python update_rankings.py --time-window 7

# Apply database migrations
python run_migrations.py
```

### Redis Cache Keys

```
search:{query_hash}          → Search results          TTL: 5 min
ranking:{resource_id}:{date} → Ranking scores          TTL: 1 hour
resource:{resource_id}       → Resource metadata       TTL: 15 min
embedding:{text_hash}        → Titan embeddings        TTL: 24 hours
```

### Useful SQL Queries

```sql
-- Top 10 trending resources
SELECT name, type, trending_score, category_rank
FROM resources
WHERE trending_score > 0.5
ORDER BY trending_score DESC
LIMIT 10;

-- Top 5 APIs by category rank
SELECT name, category_rank, rank_score
FROM resources
WHERE type = 'api'
ORDER BY category_rank ASC
LIMIT 5;

-- Most downloaded boilerplates
SELECT name, type, boilerplate_download_count
FROM resources
ORDER BY boilerplate_download_count DESC
LIMIT 10;
```

---

## 8. Troubleshooting

### "Backend offline — showing demo data"

The Next.js app handles this gracefully and falls back to `MOCK_TOOLS`. To resolve:

```bash
# 1. Start the FastAPI backend
cd backend && uvicorn main:app --reload --port 8000

# 2. Verify health
curl http://localhost:8000/api/v1/health

# 3. Check the Next.js API route resolves
curl http://localhost:3000/api/health
```

### "NEXT_PUBLIC_ variable undefined"

Only variables starting with `NEXT_PUBLIC_` are available client-side. Server-only vars (AWS, OpenSearch) must be accessed inside Route Handlers (`app/api/*/route.ts`).

### "Trending scores are all 0"

Ranking isn't computed in real time — run the update script:
```bash
cd backend && python update_rankings.py --time-window 7
```

### "Search returns no results"

- Confirm OpenSearch is running and accessible at `OPENSEARCH_HOST`
- Confirm the index exists: `curl https://<host>/devstore_resources`
- Run the ingestion pipeline to seed data

### Backend Won't Start

```bash
# Check if port 8000 is occupied
netstat -ano | findstr ":8000"   # Windows
lsof -i :8000                    # Linux/macOS

# Check Python environment
cd backend && python -c "import fastapi; print(fastapi.__version__)"
```

### Performance Tuning

| Setting | File | Recommendation |
|---------|------|----------------|
| Gunicorn workers | `start_server.sh` | `(2 × CPU cores) + 1` |
| DB pool size | `backend/.env` | `DB_POOL_SIZE=20` |
| Redis pool | `backend/.env` | `REDIS_POOL_SIZE=50` |
| ISR cache (trending) | `app/api/trending/route.ts` | `next: { revalidate: 60 }` |

---

## Project Structure

```
nextjs-frontend/
├── app/
│   ├── layout.tsx              ← SEO metadata, root layout
│   ├── page.tsx                ← Renders DevStoreDashboard
│   ├── globals.css             ← All glassmorphism tokens + keyframes
│   └── api/
│       ├── search/route.ts     ← Secure proxy: POST /api/search
│       ├── trending/route.ts   ← Secure proxy: GET  /api/trending
│       ├── resources/route.ts  ← Secure proxy: GET  /api/resources
│       └── health/route.ts     ← Secure proxy: GET  /api/health
├── components/
│   ├── DevStoreDashboard.jsx   ← 1,400-line main dashboard
│   ├── Tooltip.tsx             ← Glassmorphic tooltip
│   └── hooks/
│       └── useWindowSize.ts    ← SSR-safe responsive hook
├── lib/
│   └── api.ts                  ← Client-side fetcher → /api/*
├── public/
│   └── logo.png
├── .env.local                  ← All secrets (gitignored)
├── next.config.ts
├── setup-frontend.bat          ← Windows first-run script
└── setup-frontend.sh           ← Linux/macOS first-run script
```

---

> **Built for AI for Bharat** · Next.js 14 · TypeScript · AWS Bedrock · OpenSearch  
> Version 2.0.0 — March 2026
