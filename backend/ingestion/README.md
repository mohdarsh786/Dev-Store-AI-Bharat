# DevStore AWS-Native Ingestion Pipeline

## Quick Start (No AWS Required)

Test the ingestion pipeline locally without any AWS infrastructure:

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Test HTTP fetchers
.venv\Scripts\python.exe ingestion\test_http_fetchers.py

# 3. Run full ingestion
.venv\Scripts\python.exe ingestion\run_ingestion.py
```

**No authentication required!** All APIs work without tokens.

## Data Sources & Methods

| Source | Method | Auth Required | What It Fetches |
|--------|--------|---------------|-----------------|
| HuggingFace | HTTP API | ❌ No | ML models & datasets |
| OpenRouter | HTTP API | ❌ No | LLM models with pricing |
| GitHub | HTTP API | ⚠️ Optional | Repositories & tools |
| RapidAPI | Scrapy Crawler | ❌ No | API marketplace (web scraping) |

**Why HTTP for most sources?**
- Faster and simpler than Scrapy for REST APIs
- Direct API calls with `httpx` library
- Better error handling and rate limiting
- Only use Scrapy for RapidAPI (no official API)

**Authentication:**
- HuggingFace: No auth required (public API)
- OpenRouter: No auth required (public API)
- GitHub: Optional token (60→5000 req/hour)
- RapidAPI: No API available (web scraping)

## Architecture Overview

This directory contains the production-grade AWS-native automated data ingestion and semantic indexing pipeline for DevStore.

```
External Sources (GitHub / HuggingFace / RapidAPI)
    ↓
Scraper Workers (Scrapy on ECS Fargate)
    ↓
Deduplication Layer (ElastiCache Redis)
    ↓
Staging Queue (Amazon SQS)
    ↓
Batch Processing Worker (FastAPI on ECS)
    ↓
Embedding Generation (Amazon Bedrock)
    ↓
Aurora PostgreSQL (metadata storage)
    ↓
Amazon OpenSearch (semantic search index)
    ↓
ElastiCache Redis (query caching)
    ↓
DevStore API
```

## Components

### 1. Scheduling & Orchestration
- **EventBridge**: Triggers ingestion every 25 minutes
- **Redis Lock**: Prevents overlapping runs using `devstore:scraper_lock`

### 2. Scraping Layer (ECS Fargate)
- **Scrapy Spiders**: GitHubResourceSpider, HuggingFaceResourceSpider, RapidAPIResourceSpider
- **Output**: Normalized JSON resources

### 3. Deduplication (ElastiCache Redis)
- **Hash Generation**: SHA256(source + name + source_url)
- **Storage**: Redis set `devstore:unique_hashes`

### 4. Staging Queue (Amazon SQS)
- **Queue**: `devstore-resource-ingestion-queue`
- **Long Polling**: 20 seconds

### 5. Batch Processing Worker (FastAPI on ECS)
- **Batch Size**: 10-20 resources
- **Processing**: Normalize → Embed → Store → Index → Cache

### 6. Embedding Generation (Amazon Bedrock)
- **Model**: Titan Embeddings or Cohere Embed
- **Input**: name + description + tags
- **Dimensions**: 384-1024

### 7. Database Storage (Aurora PostgreSQL)
- **Tables**: resources, embeddings, ingestion_logs
- **Strategy**: UPSERT on conflict

### 8. Search Indexing (Amazon OpenSearch)
- **Index**: `dev-store-resources`
- **Features**: KNN vector search, semantic search

### 9. Cache Layer (ElastiCache Redis)
- **Keys**: search:*, trending:*, ranking:*
- **Invalidation**: On resource updates

### 10. Monitoring (CloudWatch)
- **Metrics**: scrape_duration, records_fetched, records_inserted, worker_errors

## Directory Structure

```
ingestion/
├── README.md                    # This file
├── config.py                    # Ingestion configuration
├── scrapers/                    # Scrapy spiders
│   ├── __init__.py
│   ├── settings.py              # Scrapy settings
│   ├── github_spider.py         # GitHub scraper
│   ├── huggingface_spider.py    # HuggingFace scraper
│   └── rapidapi_spider.py       # RapidAPI scraper
├── workers/                     # Processing workers
│   ├── __init__.py
│   ├── batch_processor.py       # Main batch processing worker
│   ├── deduplicator.py          # Deduplication logic
│   └── embedder.py              # Embedding generation
├── services/                    # Business logic
│   ├── __init__.py
│   ├── sqs_service.py           # SQS operations
│   ├── storage_service.py       # Database operations
│   └── indexing_service.py      # OpenSearch operations
├── docker/                      # Docker configurations
│   ├── scraper.Dockerfile       # Scraper container
│   └── worker.Dockerfile        # Worker container
├── terraform/                   # Infrastructure as Code
│   ├── main.tf                  # Main Terraform config
│   ├── eventbridge.tf           # EventBridge rules
│   ├── ecs.tf                   # ECS tasks and services
│   ├── sqs.tf                   # SQS queues
│   └── iam.tf                   # IAM roles and policies
└── monitoring/                  # Monitoring and logging
    ├── cloudwatch_metrics.py    # CloudWatch metrics
    └── dashboard.json           # CloudWatch dashboard
```

## Deployment

### Prerequisites
- AWS CLI configured
- Docker installed
- Terraform installed
- Python 3.11+

### Steps

1. **Build Docker Images**
```bash
cd backend/ingestion/docker
docker build -f scraper.Dockerfile -t devstore-scraper:latest .
docker build -f worker.Dockerfile -t devstore-worker:latest .
```

2. **Push to ECR**
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag devstore-scraper:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/devstore-scraper:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/devstore-scraper:latest
```

3. **Deploy Infrastructure**
```bash
cd backend/ingestion/terraform
terraform init
terraform plan
terraform apply
```

4. **Verify Deployment**
```bash
# Check ECS tasks
aws ecs list-tasks --cluster devstore-ingestion

# Check SQS queue
aws sqs get-queue-attributes --queue-url <queue-url> --attribute-names All

# Check CloudWatch logs
aws logs tail /aws/ecs/devstore-scraper --follow
```

## Testing

```bash
# Test API connectivity
python backend/ingestion/test_apis.py

# Run unit tests
pytest backend/ingestion/tests/

# Test individual spider locally
python -m scrapy crawl github_resource -o output.json
python -m scrapy crawl huggingface_resource -o output.json
python -m scrapy crawl openrouter -o output.json
python -m scrapy crawl rapidapi_resource -o output.json

# Run all spiders
python backend/ingestion/run_spiders.py

# Test worker locally
python backend/ingestion/workers/batch_processor.py
```

## Monitoring

- **CloudWatch Dashboard**: View real-time metrics
- **CloudWatch Alarms**: Alert on errors and performance issues
- **Ingestion Logs Table**: Track ingestion history

## Configuration

Environment variables (set in ECS task definitions):

```bash
# Database
DATABASE_URL=postgresql://user:pass@aurora-endpoint:5432/devstore

# Redis
REDIS_HOST=elasticache-endpoint
REDIS_PORT=6379

# OpenSearch
OPENSEARCH_HOST=opensearch-endpoint
OPENSEARCH_PORT=443

# SQS
SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/account/devstore-resource-ingestion-queue

# Bedrock
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1

# API Keys
INGESTION_GITHUB_API_TOKEN=your_github_token
INGESTION_HUGGINGFACE_API_TOKEN=your_huggingface_token
INGESTION_OPENROUTER_API_KEY=your_openrouter_key
INGESTION_RAPIDAPI_KEY=your_rapidapi_key

# AWS
AWS_REGION=us-east-1
```

## API Integration

### HuggingFace API
- Fetches ML models and datasets
- Requires API token (optional but recommended for higher rate limits)
- Get token: https://huggingface.co/settings/tokens

### OpenRouter API
- Fetches available LLM models with pricing
- Requires API key
- Get key: https://openrouter.ai/keys

### GitHub API
- Searches for repositories, APIs, and ML frameworks
- Requires personal access token for higher rate limits
- Get token: https://github.com/settings/tokens

### RapidAPI
- Scrapes API marketplace for public APIs
- Optional API key for better access
- Get key: https://rapidapi.com/developer/security

## Troubleshooting

### Scraper Issues
- Check CloudWatch logs: `/aws/ecs/devstore-scraper`
- Verify API rate limits
- Check Redis lock status

### Worker Issues
- Check SQS queue depth
- Verify Bedrock quotas
- Check database connection pool

### Performance Issues
- Scale ECS tasks
- Increase SQS batch size
- Optimize database queries
