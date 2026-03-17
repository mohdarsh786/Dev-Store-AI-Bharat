# Complete Project Setup Guide

## 🎯 Overview

This guide will help you set up the complete DevStore project in phases, from a working demo to full production.

## Phase 1: Quick Start (Working System in 15 Minutes) ✅

Get a working system with real data (2,240 resources) without any infrastructure setup.

### Step 1.1: Install Dependencies

```bash
# Backend dependencies
cd backend
pip install -r requirements.txt

# Frontend dependencies
cd ../frontend
npm install
```

### Step 1.2: Configure Environment

```bash
# Backend - Create .env file
cd backend
cp .env.example .env

# Edit .env - Minimal config for JSON mode:
# (No database, Redis, or AWS needed yet)
```

**backend/.env** (minimal):
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Mode
USE_MOCK_DATA=false
```

### Step 1.3: Start Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Test it:**
```bash
curl http://localhost:8000/api/resources/stats
```

**Expected response:**
```json
{
  "total_resources": 2240,
  "models": 1446,
  "datasets": 140,
  "repositories": 654
}
```

### Step 1.4: Configure Frontend

```bash
cd frontend

# Create .env.local
echo "BACKEND_URL=http://localhost:8000" > .env.local
```

### Step 1.5: Start Frontend

```bash
cd frontend
npm run dev
```

**Expected output:**
```
  ▲ Next.js 14.x.x
  - Local:        http://localhost:3000
```

### Step 1.6: Test the System

Open browser: **http://localhost:3000**

✅ You should see:
- Real resources (not mocks)
- Search working
- Trending resources
- 2,240 total resources

**Test search:**
- Search for "gpt" → Should return GPT models
- Search for "dataset" → Should return datasets
- Check trending → Should show popular resources

---

## Phase 2: Local Infrastructure Setup (Optional - 1 hour)

Add PostgreSQL, Redis, and OpenSearch locally for full features.

### Step 2.1: Install Docker Desktop

Download and install Docker Desktop for Windows:
https://www.docker.com/products/docker-desktop/

### Step 2.2: Create Docker Compose File

Create `docker-compose.yml` in project root:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: devstore
      POSTGRES_USER: devstore
      POSTGRES_PASSWORD: devstore123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  opensearch:
    image: opensearchproject/opensearch:2.11.0
    environment:
      - discovery.type=single-node
      - OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m
      - DISABLE_SECURITY_PLUGIN=true
    ports:
      - "9200:9200"
    volumes:
      - opensearch_data:/usr/share/opensearch/data

volumes:
  postgres_data:
  redis_data:
  opensearch_data:
```

### Step 2.3: Start Infrastructure

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Step 2.4: Update Backend Configuration

Update `backend/.env`:

```bash
# Database
DATABASE_URL=postgresql://devstore:devstore123@localhost:5432/devstore
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# OpenSearch
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USE_SSL=false
OPENSEARCH_INDEX_NAME=devstore_resources

# AWS Bedrock (Optional - for embeddings)
# AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID=your-key
# AWS_SECRET_ACCESS_KEY=your-secret
```

### Step 2.5: Run Database Migrations

```bash
cd backend
python run_migrations.py
```

**Expected output:**
```
Running migration: 001_create_resources_table.sql
✓ Migration completed
Running migration: 002_create_categories_tables.sql
✓ Migration completed
...
All migrations completed successfully!
```

### Step 2.6: Verify Infrastructure

```bash
# Test PostgreSQL
psql postgresql://devstore:devstore123@localhost:5432/devstore -c "SELECT 1;"

# Test Redis
redis-cli PING

# Test OpenSearch
curl http://localhost:9200/_cluster/health
```

---

## Phase 3: Run Full Ingestion Pipeline (30 minutes)

Populate the database with real data using the production pipeline.

### Step 3.1: Verify Infrastructure

```bash
cd backend
python scripts/test_pipeline_dry_run.py
```

**Expected output:**
```
✅ DRY RUN SUCCESSFUL!
All components are working correctly.
```

### Step 3.2: Run Initial Backfill

**Option A: Without AWS Bedrock (No embeddings)**

```bash
cd backend

# Run ingestion in JSON mode (no embeddings)
python -c "
import asyncio
from ingestion.pipeline import run_ingestion

async def main():
    result = await run_ingestion(sources=['github', 'huggingface'])
    print(f'Status: {result[\"status\"]}')
    print(f'Inserted: {result[\"stats\"][\"inserted\"]}')

asyncio.run(main())
"
```

**Option B: With AWS Bedrock (Full features)**

First, configure AWS credentials in `.env`:
```bash
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1
```

Then run:
```bash
python scripts/run_full_backfill.py
```

**Expected output:**
```
======================================================================
FULL BACKFILL STARTED
======================================================================
Verifying infrastructure...
✓ Database healthy
✓ Redis healthy
✓ OpenSearch healthy

Running ingestion across all sources...
Stage 1: Fetch & Normalize
✓ GitHub: 654 repositories
✓ HuggingFace: 1000 models, 100 datasets
✓ OpenRouter: 346 models
✓ Kaggle: 40 datasets

Stage 2: Deduplicate
Resources: 2140 → 2140 (0 duplicates)

Stage 3: Upsert to PostgreSQL
Inserted: 2140, Updated: 0

Stage 4: Generate Embeddings
Embeddings generated: 2140

Stage 5: Index in OpenSearch
Resources indexed: 2140

Stage 6: Refresh Rankings
✓ Rankings updated: 2140 resources

Stage 7: Invalidate Caches
✓ Caches invalidated

======================================================================
✅ BACKFILL COMPLETED SUCCESSFULLY
======================================================================
```

### Step 3.3: Verify Data

```bash
# Check database
psql postgresql://devstore:devstore123@localhost:5432/devstore -c "SELECT COUNT(*) FROM resources;"

# Check OpenSearch
curl "http://localhost:9200/devstore_resources/_count"

# Check latest ingestion run
psql postgresql://devstore:devstore123@localhost:5432/devstore -c "SELECT * FROM ingestion_runs ORDER BY started_at DESC LIMIT 1;"
```

---

## Phase 4: Schedule Automatic Updates (15 minutes)

Set up automatic data refresh every 6 hours.

### Step 4.1: Create Scheduled Task (Windows)

Create `run_ingestion.bat`:

```batch
@echo off
cd C:\path\to\Dev-Store-AI-Bharat\backend
call .venv\Scripts\activate
python scripts/run_full_backfill.py >> logs\ingestion.log 2>&1
```

### Step 4.2: Set Up Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Name: "DevStore Ingestion"
4. Trigger: Daily, repeat every 6 hours
5. Action: Start a program
6. Program: `C:\path\to\run_ingestion.bat`
7. Finish

### Step 4.3: Test Scheduled Task

```bash
# Run manually to test
run_ingestion.bat

# Check logs
type logs\ingestion.log
```

---

## Phase 5: Production Deployment (Optional)

Deploy to AWS EC2 for production use.

### Step 5.1: Set Up AWS Infrastructure

Follow `AWS_SETUP_GUIDE.md` to create:
- RDS PostgreSQL instance
- ElastiCache Redis cluster
- OpenSearch domain
- S3 bucket for snapshots
- IAM roles and policies

### Step 5.2: Deploy to EC2

Follow `EC2_DEPLOYMENT_GUIDE.md` to:
- Launch EC2 instance
- Install dependencies
- Configure systemd services
- Set up Nginx reverse proxy
- Configure SSL certificates

---

## 🎯 Quick Reference

### Start Development Environment

```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev

# Terminal 3: Infrastructure (if using Docker)
docker-compose up
```

### Stop Everything

```bash
# Stop frontend/backend: Ctrl+C in terminals

# Stop Docker infrastructure
docker-compose down
```

### Useful Commands

```bash
# Check API health
curl http://localhost:8000/api/resources/stats

# Check database
psql postgresql://devstore:devstore123@localhost:5432/devstore -c "SELECT COUNT(*) FROM resources;"

# Check Redis
redis-cli PING

# Check OpenSearch
curl http://localhost:9200/_cluster/health

# View backend logs
tail -f backend/ingestion/logs/ingestion_*.log

# Run tests
cd backend
pytest tests/test_ingestion_pipeline.py -v
```

---

## 🐛 Troubleshooting

### Backend won't start

```bash
# Check Python version (need 3.11+)
python --version

# Reinstall dependencies
cd backend
pip install -r requirements.txt --force-reinstall
```

### Frontend won't start

```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules .next
npm install
npm run dev
```

### Database connection fails

```bash
# Check Docker is running
docker-compose ps

# Restart PostgreSQL
docker-compose restart postgres

# Check connection
psql postgresql://devstore:devstore123@localhost:5432/devstore -c "SELECT 1;"
```

### Ingestion fails

```bash
# Check logs
tail -f backend/ingestion/logs/ingestion_*.log

# Check infrastructure
python scripts/test_pipeline_dry_run.py

# Run with debug logging
cd backend
python scripts/run_full_backfill.py --log-level DEBUG
```

---

## 📊 What You'll Have After Setup

### Phase 1 Complete:
- ✅ Working frontend and backend
- ✅ 2,240 real resources
- ✅ Search functionality
- ✅ Trending resources
- ✅ No infrastructure needed

### Phase 2 Complete:
- ✅ PostgreSQL database
- ✅ Redis caching
- ✅ OpenSearch indexing
- ✅ All running locally

### Phase 3 Complete:
- ✅ Database populated with resources
- ✅ Embeddings generated (if using Bedrock)
- ✅ Full-text search working
- ✅ Rankings computed

### Phase 4 Complete:
- ✅ Automatic data refresh every 6 hours
- ✅ Scheduled ingestion
- ✅ Logs and monitoring

### Phase 5 Complete:
- ✅ Production deployment on AWS
- ✅ Scalable infrastructure
- ✅ SSL certificates
- ✅ Domain name

---

## 🎉 Success Criteria

After Phase 1, you should be able to:
- ✅ Open http://localhost:3000
- ✅ Search for "gpt" and see real models
- ✅ View trending resources
- ✅ See 2,240 total resources

After Phase 3, you should be able to:
- ✅ Query PostgreSQL and see resources
- ✅ Search in OpenSearch
- ✅ See embeddings in database
- ✅ View ranking scores

---

## 📞 Need Help?

- Check logs: `backend/ingestion/logs/`
- Run dry run test: `python scripts/test_pipeline_dry_run.py`
- Check documentation: `backend/ingestion/PRODUCTION_PIPELINE.md`
- Review verification: `PIPELINE_VERIFICATION.md`
