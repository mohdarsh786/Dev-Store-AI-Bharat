# Production Ingestion Pipeline

## Overview

Complete production-ready ingestion pipeline that fetches, normalizes, deduplicates, stores, embeds, indexes, and ranks resources from multiple external sources.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Ingestion Pipeline                         │
│                                                              │
│  1. Acquire Lock (Redis)                                    │
│  2. Fetch (GitHub, HuggingFace, Kaggle, OpenRouter)        │
│  3. Normalize (Canonical Schema)                            │
│  4. Deduplicate (source + source_url)                       │
│  5. Upsert (PostgreSQL)                                     │
│  6. Generate Embeddings (Bedrock + Redis Cache)             │
│  7. Index (OpenSearch)                                       │
│  8. Refresh Rankings                                         │
│  9. Invalidate Caches (Redis)                               │
│  10. Release Lock                                            │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Pipeline Stages

Located in `ingestion/stages/`:

- **FetchStage** - Retrieves data from external sources
- **NormalizeStage** - Converts to canonical schema
- **DedupeStage** - Removes duplicates by unique_key
- **UpsertStage** - Inserts/updates in PostgreSQL
- **EmbeddingStage** - Generates embeddings via Bedrock
- **IndexingStage** - Indexes in OpenSearch
- **RankingStage** - Computes ranking scores
- **CacheStage** - Invalidates Redis caches

### Core Services

- **LockService** - Distributed locking via Redis
- **RunTracker** - Tracks ingestion run status
- **SnapshotStore** - Persists raw payloads to S3

### Repository

- **IngestionRepository** - Database operations with upsert logic

## Usage

### Running Full Backfill

```bash
cd backend
python scripts/run_full_backfill.py
```

This will:
1. Verify infrastructure (PostgreSQL, Redis, OpenSearch, Bedrock)
2. Run ingestion across all sources
3. Verify results (records, embeddings, indexing, rankings)

### Running Specific Sources

```python
from ingestion.pipeline import run_ingestion

# Run specific sources
result = await run_ingestion(sources=["github", "huggingface"])

# Run all sources
result = await run_ingestion()
```

### Programmatic Usage

```python
from ingestion.pipeline import IngestionPipeline
from clients.database import DatabaseClient
from clients.bedrock import BedrockClient
from clients.opensearch import OpenSearchClient
from clients.redis_client import RedisClient

# Initialize clients
db = DatabaseClient()
bedrock = BedrockClient()
opensearch = OpenSearchClient()
redis = RedisClient()

# Create pipeline
pipeline = IngestionPipeline(
    db_client=db,
    bedrock_client=bedrock,
    opensearch_client=opensearch,
    redis_client=redis,
)

# Run pipeline
result = await pipeline.run(sources=["github", "huggingface"])

print(f"Status: {result['status']}")
print(f"Inserted: {result['stats']['inserted']}")
print(f"Updated: {result['stats']['updated']}")
print(f"Embedded: {result['stats']['embedded']}")
print(f"Indexed: {result['stats']['indexed']}")
```

## Scheduling

### Systemd Timer (Recommended for EC2)

```bash
# Copy service files
sudo cp systemd/devstore-full-ingestion.* /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable timer
sudo systemctl enable devstore-full-ingestion.timer

# Start timer
sudo systemctl start devstore-full-ingestion.timer

# Check status
sudo systemctl status devstore-full-ingestion.timer

# View logs
sudo journalctl -u devstore-full-ingestion.service -f
```

### Cron (Alternative)

```bash
# Edit crontab
crontab -e

# Add entry (runs every 6 hours)
0 */6 * * * cd /opt/devstore/backend && /opt/devstore/backend/.venv/bin/python scripts/run_full_backfill.py >> /var/log/devstore-ingestion.log 2>&1
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/devstore

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# AWS Bedrock
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret

# OpenSearch
OPENSEARCH_HOST=your-endpoint.amazonaws.com

# S3 (for snapshots)
S3_BUCKET_CRAWLER_DATA=devstore-crawler-data
CRAWLER_SNAPSHOT_PREFIX=snapshots
```

## Canonical Resource Schema

All resources are normalized to this schema:

```python
CanonicalResource:
    source: IngestionSource  # github, huggingface, kaggle, openrouter
    resource_type: ResourceType  # api, model, dataset
    name: str
    description: str
    source_url: str
    documentation_url: Optional[str]
    pricing_type: PricingType  # free, paid, freemium
    github_stars: int
    download_count: int
    active_users: int
    health_status: HealthStatus
    tags: List[str]
    categories: List[str]
    metadata: Dict[str, Any]
    source_updated_at: Optional[datetime]
    raw_payload: Dict[str, Any]
```

## Deduplication

Resources are deduplicated using:

```python
unique_key = source + source_url
```

Example:
- `github:https://github.com/user/repo`
- `huggingface:https://huggingface.co/model/name`

## Change Detection

Resources are updated only if content changed:

```python
content_hash = sha256(
    source + resource_type + name + description +
    tags + metadata + ...
)
```

If `content_hash` matches existing record, update is skipped.

## Idempotency

The pipeline is fully idempotent:

- Rerunning does NOT create duplicate Aurora rows
- Rerunning does NOT create duplicate OpenSearch documents
- Unchanged resources are skipped
- Changed resources trigger:
  - Database update
  - Embedding regeneration
  - OpenSearch reindex

## Embedding Caching

Embeddings are cached in Redis for 30 days:

```python
cache_key = f"embedding:{sha256(embedding_text)}"
ttl = 86400 * 30  # 30 days
```

This significantly reduces Bedrock costs on reruns.

## Distributed Locking

Redis lock prevents overlapping runs:

```python
lock_key = "ingestion:lock"
lock_ttl = 3600  # 1 hour
```

If lock is held, pipeline exits with status "skipped".

## Run Tracking

Each ingestion run is tracked in database:

```sql
CREATE TABLE ingestion_runs (
    run_id UUID,
    source VARCHAR,
    status VARCHAR,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    fetched_count INT,
    inserted_count INT,
    updated_count INT,
    failed_count INT,
    ...
);
```

Query latest run:

```python
from ingestion.services.run_tracker import RunTracker

tracker = RunTracker(db_client)
latest = tracker.get_latest_run_status(source="github")
```

## Snapshot Storage

Raw payloads are persisted to S3:

```
s3://devstore-crawler-data/
  snapshots/
    github/
      2024/03/07/120000/
        run-id.json
    huggingface/
      2024/03/07/120000/
        run-id.json
```

Purpose:
- Debugging
- Replay
- Audit trail

## Monitoring

### Logs

Logs are written to:
- Console (INFO level)
- File: `ingestion/logs/ingestion_{run_id}.log` (DEBUG level)
- Systemd journal (if using systemd)

### Metrics

Each run tracks:

```python
{
    "fetched": 2040,
    "normalized": 2040,
    "deduplicated": 2040,
    "inserted": 150,
    "updated": 1890,
    "unchanged": 0,
    "embedded": 150,
    "indexed": 2040,
    "ranked": 2040,
    "failed": 0
}
```

### Health Checks

```bash
# Check infrastructure
python scripts/run_full_backfill.py --check-only

# Check latest run
psql -c "SELECT * FROM ingestion_runs ORDER BY started_at DESC LIMIT 1;"

# Check Redis lock
redis-cli GET ingestion:lock
```

## Testing

### Run Integration Tests

```bash
cd backend
pytest tests/test_ingestion_pipeline.py -v
```

Tests cover:
1. Fetch → Normalize → Upsert → Index flow
2. Idempotency (reruns don't duplicate)
3. Update detection (changed resources trigger updates)
4. Redis lock behavior
5. Embedding cache hits
6. Cache invalidation
7. Operational health
8. Partial failure handling

### Manual Testing

```bash
# Test with single source
python -c "
import asyncio
from ingestion.pipeline import run_ingestion

result = asyncio.run(run_ingestion(sources=['github']))
print(result)
"

# Test full pipeline
python scripts/run_full_backfill.py
```

## Troubleshooting

### Lock Already Held

If another ingestion is running:

```bash
# Check lock
redis-cli GET ingestion:lock

# Force release (if needed)
redis-cli DEL ingestion:lock
```

### Embedding Generation Fails

Check Bedrock access:

```bash
aws bedrock list-foundation-models --region us-east-1
```

### OpenSearch Indexing Fails

Check cluster health:

```bash
curl -X GET "https://your-endpoint.amazonaws.com/_cluster/health"
```

### Database Connection Issues

Check PostgreSQL:

```bash
psql $DATABASE_URL -c "SELECT 1;"
```

## Performance

### Throughput

- Fetching: ~100 resources/second
- Normalization: ~1000 resources/second
- Database upsert: ~50 resources/second
- Embedding generation: ~10 resources/second (Bedrock limit)
- OpenSearch indexing: ~100 resources/second (bulk)

### Optimization Tips

1. **Batch Size**: Adjust batch sizes in stages
2. **Parallel Fetching**: Fetch sources in parallel (future enhancement)
3. **Embedding Cache**: Ensure Redis is available for caching
4. **OpenSearch Bulk**: Use bulk API for indexing

## Cost Estimation

### Per Run (10,000 resources)

- Bedrock Embeddings: $5-10 (with caching: $0.50-1)
- OpenSearch: Included in cluster cost
- RDS: Included in instance cost
- Redis: Included in node cost
- S3: ~$0.01 for snapshots

### Monthly (4 runs/day)

- Bedrock: $60-120 (with caching: $6-12)
- OpenSearch: $50-200
- RDS: $50-150
- Redis: $20-50
- S3: ~$1

**Total: ~$126-520/month** (with caching: ~$127-413/month)

## Future Enhancements

### SQS-Based Scaling

For high-volume ingestion:

```python
# Producer
for source in sources:
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps({"source": source, "run_id": run_id})
    )

# Worker
while True:
    messages = sqs.receive_message(QueueUrl=queue_url)
    for message in messages:
        job = json.loads(message["Body"])
        process_source(job["source"], job["run_id"])
```

### Parallel Source Fetching

```python
async def fetch_all_sources(sources):
    tasks = [fetch_source(source) for source in sources]
    return await asyncio.gather(*tasks)
```

### Incremental Updates

Track last successful run per source:

```python
last_run = tracker.get_latest_run_status(source="github")
if last_run:
    fetch_since = last_run["finished_at"]
```

## Support

For issues:
1. Check logs in `ingestion/logs/`
2. Verify infrastructure health
3. Check Redis lock status
4. Review run tracking table
5. Examine S3 snapshots for debugging

## Summary

✅ Complete production pipeline implemented
✅ All 10 stages working
✅ Idempotent and safe to rerun
✅ Distributed locking prevents overlaps
✅ Embedding caching reduces costs
✅ Change detection skips unchanged resources
✅ Comprehensive testing
✅ Scheduled execution via systemd/cron
✅ Full monitoring and logging
✅ Ready for production deployment
