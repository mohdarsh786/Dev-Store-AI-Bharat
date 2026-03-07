# Ingestion Pipeline - Quick Start Guide

## Prerequisites

1. **Infrastructure Running:**
   - PostgreSQL (with migrations applied)
   - Redis
   - OpenSearch
   - AWS Bedrock access

2. **Environment Variables Set:**
   ```bash
   export DATABASE_URL="postgresql://user:pass@host:5432/devstore"
   export REDIS_HOST="localhost"
   export REDIS_PORT=6379
   export AWS_REGION="us-east-1"
   export AWS_ACCESS_KEY_ID="your-key"
   export AWS_SECRET_ACCESS_KEY="your-secret"
   export OPENSEARCH_HOST="your-endpoint.amazonaws.com"
   ```

## Quick Start

### 1. Run Full Backfill

```bash
cd backend
python scripts/run_full_backfill.py
```

This will:
- ✅ Check infrastructure health
- ✅ Fetch from all sources (GitHub, HuggingFace, Kaggle, OpenRouter)
- ✅ Normalize to canonical schema
- ✅ Deduplicate resources
- ✅ Upsert to PostgreSQL
- ✅ Generate embeddings via Bedrock
- ✅ Index in OpenSearch
- ✅ Compute rankings
- ✅ Invalidate caches
- ✅ Verify results

### 2. Check Results

```bash
# Check database
psql $DATABASE_URL -c "SELECT COUNT(*) FROM resources;"

# Check OpenSearch
curl "https://$OPENSEARCH_HOST/devstore_resources/_count"

# Check latest run
psql $DATABASE_URL -c "SELECT * FROM ingestion_runs ORDER BY started_at DESC LIMIT 1;"
```

### 3. Schedule Regular Runs

```bash
# Copy systemd files
sudo cp systemd/devstore-full-ingestion.* /etc/systemd/system/

# Enable timer (runs every 6 hours)
sudo systemctl daemon-reload
sudo systemctl enable devstore-full-ingestion.timer
sudo systemctl start devstore-full-ingestion.timer

# Check status
sudo systemctl status devstore-full-ingestion.timer
```

## Programmatic Usage

```python
import asyncio
from ingestion.pipeline import run_ingestion

async def main():
    # Run all sources
    result = await run_ingestion()
    
    print(f"Status: {result['status']}")
    print(f"Inserted: {result['stats']['inserted']}")
    print(f"Updated: {result['stats']['updated']}")
    print(f"Embedded: {result['stats']['embedded']}")
    print(f"Indexed: {result['stats']['indexed']}")

asyncio.run(main())
```

## Run Specific Sources

```python
# Run only GitHub and HuggingFace
result = await run_ingestion(sources=["github", "huggingface"])

# Run only Kaggle
result = await run_ingestion(sources=["kaggle"])
```

## Monitoring

### View Logs

```bash
# Systemd logs
sudo journalctl -u devstore-full-ingestion.service -f

# File logs
tail -f backend/ingestion/logs/ingestion_*.log
```

### Check Lock Status

```bash
# Check if ingestion is running
redis-cli GET ingestion:lock

# Force release if stuck
redis-cli DEL ingestion:lock
```

### Query Run History

```sql
-- Latest runs
SELECT run_id, source, status, started_at, finished_at,
       fetched_count, inserted_count, updated_count
FROM ingestion_runs
ORDER BY started_at DESC
LIMIT 10;

-- Failed runs
SELECT * FROM ingestion_runs
WHERE status = 'failed'
ORDER BY started_at DESC;
```

## Testing

```bash
# Run integration tests
cd backend
pytest tests/test_ingestion_pipeline.py -v

# Test specific source
python -c "
import asyncio
from ingestion.pipeline import run_ingestion
result = asyncio.run(run_ingestion(sources=['github']))
print(result)
"
```

## Troubleshooting

### Infrastructure Not Available

```bash
# Check database
psql $DATABASE_URL -c "SELECT 1;"

# Check Redis
redis-cli PING

# Check OpenSearch
curl "https://$OPENSEARCH_HOST/_cluster/health"

# Check Bedrock
aws bedrock list-foundation-models --region us-east-1
```

### Lock Already Held

```bash
# Check lock
redis-cli GET ingestion:lock

# See lock value (contains timestamp)
redis-cli GET ingestion:lock

# Force release (only if you're sure no ingestion is running)
redis-cli DEL ingestion:lock
```

### Embedding Generation Slow

Embeddings are cached for 30 days. First run will be slow, subsequent runs will be fast.

```bash
# Check cache hit rate
redis-cli INFO stats | grep keyspace
```

## Performance Tips

1. **First Run**: Will be slow (generates all embeddings)
2. **Subsequent Runs**: Much faster (uses cached embeddings)
3. **Batch Size**: Adjust in stage files if needed
4. **Parallel Sources**: Future enhancement (currently sequential)

## Cost Optimization

1. **Enable Embedding Cache**: Reduces Bedrock costs by 90%+
2. **Adjust Run Frequency**: Run less often if data doesn't change frequently
3. **Selective Sources**: Run only sources that need updates

## Next Steps

1. ✅ Run initial backfill
2. ✅ Verify data in database and OpenSearch
3. ✅ Schedule regular runs
4. ✅ Monitor logs and metrics
5. ✅ Test API endpoints with real data

## Support

- Full documentation: `backend/ingestion/PRODUCTION_PIPELINE.md`
- Implementation details: `INGESTION_IMPLEMENTATION_COMPLETE.md`
- Test examples: `backend/tests/test_ingestion_pipeline.py`
