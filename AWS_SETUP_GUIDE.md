# AWS Setup Guide for DevStore

## Current Status: Mock Data Mode 📦

The application is currently running with **mock data**. To enable real AWS-powered search with Bedrock and OpenSearch, follow this guide.

## How It Works

The backend automatically detects if AWS services are configured:

- ✅ **AWS Configured**: Uses Bedrock (Claude + Titan) + OpenSearch for real AI-powered search
- 📦 **AWS Not Configured**: Uses mock data (current mode)

## Environment Variables Required

Create `backend/.env` with these variables:

```env
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/devstore

# OpenSearch
OPENSEARCH_HOST=your-opensearch-domain.us-east-1.es.amazonaws.com
OPENSEARCH_PORT=443
OPENSEARCH_USE_SSL=true
OPENSEARCH_INDEX_NAME=devstore_resources

# Bedrock Models
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1

# S3 Buckets
S3_BUCKET_BOILERPLATE=devstore-boilerplate
S3_BUCKET_CRAWLER_DATA=devstore-crawler

# API Settings
API_RATE_LIMIT=100
API_TIMEOUT=30
ENVIRONMENT=development
LOG_LEVEL=INFO
```

## Step-by-Step AWS Setup

### 1. AWS Account Setup

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
```

### 2. Create RDS PostgreSQL Database

```bash
# Create database
aws rds create-db-instance \
  --db-instance-identifier devstore-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username admin \
  --master-user-password YourPassword123 \
  --allocated-storage 20

# Get connection string
aws rds describe-db-instances \
  --db-instance-identifier devstore-db \
  --query 'DBInstances[0].Endpoint.Address'
```

### 3. Create OpenSearch Domain

```bash
# Create OpenSearch domain
aws opensearch create-domain \
  --domain-name devstore-search \
  --engine-version OpenSearch_2.11 \
  --cluster-config InstanceType=t3.small.search,InstanceCount=1 \
  --ebs-options EBSEnabled=true,VolumeType=gp3,VolumeSize=10

# Get endpoint
aws opensearch describe-domain \
  --domain-name devstore-search \
  --query 'DomainStatus.Endpoint'
```

### 4. Enable Bedrock Access

```bash
# Check Bedrock model access
aws bedrock list-foundation-models --region us-east-1

# Request model access (if needed)
# Go to AWS Console > Bedrock > Model access > Request access
```

### 5. Create S3 Buckets

```bash
# Create buckets
aws s3 mb s3://devstore-boilerplate
aws s3 mb s3://devstore-crawler
```

### 6. Run Database Migrations

```bash
cd backend
python run_migrations.py
```

### 7. Restart Backend

```bash
python -m uvicorn main:app --reload --port 8000
```

You should see:
```
✅ AWS services configured - using real search
```

## Verification

### Check Backend Logs

When backend starts, you'll see one of:
- `✅ AWS services configured - using real search` - AWS mode
- `📦 AWS not configured - using mock data` - Mock mode

### Test Search

```bash
# Search should return source: "aws" or "mock"
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "image generation", "limit": 5}'
```

Response will include:
```json
{
  "query": "image generation",
  "results": [...],
  "source": "aws"  // or "mock"
}
```

## Cost Estimates (AWS)

| Service | Configuration | Monthly Cost |
|---------|--------------|--------------|
| RDS PostgreSQL | db.t3.micro | ~$15 |
| OpenSearch | t3.small.search | ~$30 |
| Bedrock | Pay per use | ~$10-50 |
| S3 | Standard storage | ~$1-5 |
| **Total** | | **~$56-100/month** |

## Development Mode (Free)

For development, you can use:
- **Mock Data** (current) - Free, no AWS needed
- **LocalStack** - Local AWS emulation
- **AWS Free Tier** - Limited free usage

## Troubleshooting

### Issue: "AWS not configured"

**Solution:** Set these environment variables:
```bash
export AWS_REGION=us-east-1
export OPENSEARCH_HOST=your-domain.es.amazonaws.com
export DATABASE_URL=postgresql://...
```

### Issue: "Bedrock access denied"

**Solution:** Request model access in AWS Console:
1. Go to AWS Bedrock console
2. Click "Model access"
3. Request access to Claude 3 and Titan models

### Issue: "OpenSearch connection failed"

**Solution:** Check security group allows your IP:
```bash
aws opensearch update-domain-config \
  --domain-name devstore-search \
  --access-policies '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"*"},"Action":"es:*","Resource":"*"}]}'
```

## Quick Start (Mock Mode)

Don't want to set up AWS yet? The app works perfectly with mock data:

```bash
# Backend
cd backend
python -m uvicorn main:app --reload

# Frontend
cd frontend
npm run dev
```

No configuration needed! 🎉

## Migration Path

1. **Start**: Mock data (current)
2. **Add Database**: Set DATABASE_URL
3. **Add OpenSearch**: Set OPENSEARCH_HOST
4. **Add Bedrock**: Set AWS credentials
5. **Full AWS**: All services configured

Each step is optional and the app gracefully falls back to mock data.
