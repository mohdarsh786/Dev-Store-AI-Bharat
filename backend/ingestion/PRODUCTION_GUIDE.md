# Production Ingestion System Guide

## Overview

The production ingestion system uses existing infrastructure components to fetch, process, and index resources from multiple sources.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Orchestrator                   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│   Fetchers   │      │  Repository  │     │   Services   │
│              │      │              │     │              │
│ • HuggingFace│      │ • Upsert     │     │ • Embedding  │
│ • OpenRouter │      │ • Dedupe     │     │ • Indexing   │
│ • GitHub     │      │ • Change     │     │ • Ranking    │
│ • Kaggle     │      │   Detection  │     │ • Caching    │
└──────────────┘      └──────────────┘     └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│  PostgreSQL  │      │  OpenSearch  │     │    Redis     │
│              │      │              │     │              │
│ • Resources  │      │ • Full-text  │     │ • Caching    │
│ • Rankings   │      │ • Vector     │     │ • Locking    │
│ • Metadata   │      │   Search     │     │ • Sessions   │
└──────────────┘      └──────────────┘     └──────────────┘
```

## Components

### 1. Infrastructure Clients

Located in `backend/clients/`:

- **DatabaseClient** (`database.py`)
  - PostgreSQL connection pooling
  - Query execution with retry logic
  - Health checks

- **BedrockClient** (`bedrock.py`)
  - AWS Bedrock integration
  - Embedding generation (Titan)
  - Text generation (Claude)
  - Circuit breaker pattern

- **OpenSearchClient** (`opensearch.py`)
  - Index management
  - KNN vector search
  - Bulk indexing

- **RedisClient** (`redis_client.py`)
  - Caching layer
  - Distributed locking
  - Session management

### 2. Repository Layer

Located in `backend/ingestion/`:

- **IngestionRepository** (`repository.py`)
  - Resource upsert logic
  - Deduplication by source + URL
  - Change detection via content hash
  - Embedding management
  - Ranking persistence

### 3. Service Layer

Located in `backend/ingestion/services/`:

- **EmbeddingService** (`embedding_service.py`)
  - Generates embeddings via Bedrock
  - Redis caching for embeddings
  - Batch processing

- **IndexingService** (`indexing_service.py`)
  - OpenSearch document indexing
  - Bulk operations
  - Document ID alignment

- **RankingService** (`ranking_service.py`)
  - Computes rank_score
  - Computes trending_score
  - Category ranking

### 4. Fetchers

Located in `backend/ingestion/fetchers/`:

- **HuggingFaceFetcher** - Models and datasets from HuggingFace
- **OpenRouterFetcher** - AI models from OpenRouter
- **GitHubFetcher** - Popular repositories
- **KaggleFetcher** - Datasets from Kaggle

## Pipeline Stages

### Stage 0: Acquire Lock
- Acquires distributed lock via Redis
- Prevents concurrent ingestion runs
- TTL: 1 hour

### Stage 1: Fetch & Normalize
- Fetches data from all enabled sources
- Normalizes to canonical schema
- Tracks per-source statistics

### Stage 2: Deduplicate
- Removes duplicates by source + source_url
- Keeps first occurrence

### Stage 3: Upsert to PostgreSQL
- Inserts new resources
- Updates changed resources
- Skips unchanged resources
- Uses content hash for change detection

### Stage 4: Generate Embeddings
- Generates embeddings via Bedrock Titan
- Caches embeddings in Redis (30 days)
- Only processes changed resources

### Stage 5: Index in OpenSearch
- Bulk indexes documents
- Includes embedding vectors
- Aligns document IDs with PostgreSQL

### Stage 6: Refresh Rankings
- Computes rank_score (popularity)
- Computes trending_score (recency)
- Computes category_rank
- Persists to database

### Stage 7: Invalidate Caches
- Clears search caches
- Clears ranking caches
- Clears resource caches

### Stage 8: Release Lock
- Releases distributed lock
- Allows next run

## Running the Pipeline

### Prerequisites

1. **Infrastructure Setup**
   ```bash
   # PostgreSQL
   export DATABASE_URL="postgresql://user:pass@host:5432/devstore"
   
   # Redis
   export REDIS_HOST="localhost"
   export REDIS_PORT=6379
   
   # AWS Bedrock
   export AWS_REGION="us-east-1"
   export AWS_ACCESS_KEY_ID="your-key"
   export AWS_SECRET_ACCESS_KEY="your-secret"
   
   # OpenSearch
   export OPENSEARCH_HOST="your-opensearch-endpoint"
   ```

2. **Database Migrations**
   ```bash
   cd backend
   python run_migrations.py
   ```

### Running with Infrastructure

```bash
cd backend/ingestion
python run_production.py
```

This will:
1. Check infrastructure availability
2. Run full pipeline with PostgreSQL, Redis, Bedrock, OpenSearch
3. Generate embeddings and index in OpenSearch

### Running in JSON Mode (No Infrastructure)

```bash
cd backend/ingestion
python run_production.py --force-json
```

This will:
1. Fetch and normalize data
2. Save to JSON files in `output/`
3. Skip database, embeddings, and indexing

### Options

```bash
# Run specific sources only
python run_production.py --sources huggingface github

# Change log level
python run_production.py --log-level DEBUG

# Check infrastructure only
python run_production.py --check-only

# Force JSON mode
python run_production.py --force-json
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/devstore
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_POOL_SIZE=10

# AWS Bedrock
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1

# OpenSearch
OPENSEARCH_HOST=your-endpoint.amazonaws.com
OPENSEARCH_PORT=443
OPENSEARCH_USE_SSL=true
OPENSEARCH_INDEX_NAME=devstore_resources
```

### Config File

See `backend/config.py` for all configuration options.

## Monitoring

### Logs

Logs are written to:
- Console (INFO level)
- File: `backend/ingestion/logs/ingestion_{run_id}.log` (DEBUG level)

### Metrics

The pipeline tracks:
- `fetched_count` - Resources fetched from sources
- `inserted_count` - New resources inserted
- `updated_count` - Existing resources updated
- `skipped_count` - Duplicates skipped
- `embedded_count` - Embeddings generated
- `indexed_count` - Documents indexed
- `failed_count` - Failed operations

### Health Checks

Check infrastructure health:
```bash
python run_production.py --check-only
```

## Scheduling

### Systemd Timer (Linux)

```bash
# Copy service files
sudo cp backend/systemd/devstore-ingestion.* /etc/systemd/system/

# Enable and start timer
sudo systemctl enable devstore-ingestion.timer
sudo systemctl start devstore-ingestion.timer

# Check status
sudo systemctl status devstore-ingestion.timer
```

### Cron (Linux/Mac)

```bash
# Run every 6 hours
0 */6 * * * cd /path/to/backend/ingestion && python run_production.py >> /var/log/devstore-ingestion.log 2>&1
```

### Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (e.g., daily at 2 AM)
4. Action: Start a program
5. Program: `python`
6. Arguments: `run_production.py`
7. Start in: `C:\path\to\backend\ingestion`

## Troubleshooting

### Infrastructure Not Available

If infrastructure check fails:
1. Verify environment variables are set
2. Check network connectivity
3. Verify credentials
4. Run in JSON mode as fallback

### Lock Already Held

If another ingestion is running:
- Wait for it to complete (max 1 hour)
- Or manually release lock in Redis:
  ```bash
  redis-cli DEL ingestion:lock
  ```

### Embedding Generation Fails

If Bedrock is unavailable:
- Check AWS credentials
- Verify region supports Bedrock
- Check Bedrock service quotas
- Pipeline will continue without embeddings

### OpenSearch Indexing Fails

If OpenSearch is unavailable:
- Check endpoint and credentials
- Verify index exists
- Check OpenSearch cluster health
- Resources will still be in PostgreSQL

## Performance

### Throughput

- Fetching: ~100 resources/second
- Database upsert: ~50 resources/second
- Embedding generation: ~10 resources/second (Bedrock limit)
- OpenSearch indexing: ~100 resources/second (bulk)

### Optimization

1. **Batch Size**
   - Embeddings: 25 per batch
   - Indexing: 100 per batch

2. **Caching**
   - Embeddings cached for 30 days
   - Search results cached for 5 minutes
   - Rankings cached for 1 hour

3. **Parallel Processing**
   - Fetchers run sequentially (to avoid rate limits)
   - Embeddings can be parallelized
   - Indexing uses bulk API

## Cost Estimation

### AWS Bedrock

- Titan Embeddings: $0.0001 per 1K tokens
- For 10,000 resources: ~$5-10 per run

### OpenSearch

- Depends on cluster size
- Typical: $50-200/month for small cluster

### RDS PostgreSQL

- Depends on instance size
- Typical: $50-150/month for db.t3.medium

### ElastiCache Redis

- Depends on node size
- Typical: $20-50/month for cache.t3.micro

## Next Steps

1. **Set up infrastructure** - See `AWS_SETUP_GUIDE.md`
2. **Run migrations** - `python run_migrations.py`
3. **Test pipeline** - `python run_production.py --check-only`
4. **Run first ingestion** - `python run_production.py`
5. **Schedule regular runs** - Set up cron/systemd timer
6. **Monitor logs** - Check `logs/` directory
7. **Verify API** - Test search endpoints

## Support

For issues or questions:
1. Check logs in `backend/ingestion/logs/`
2. Verify infrastructure with `--check-only`
3. Try JSON mode with `--force-json`
4. Review error messages and stack traces
