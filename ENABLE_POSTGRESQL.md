# Enable PostgreSQL Data Pipeline

## Current Status: JSON Mode

Right now, your system works in **JSON mode**:
- ✅ Backend serves data from JSON files
- ✅ No database needed
- ✅ 2,240 resources available immediately
- ✅ Perfect for development and testing

## Data Flow Comparison

### Current: JSON Mode
```
Frontend → Backend API → JSON Files
                          └── backend/ingestion/output/*.json
```

### With PostgreSQL: Full Pipeline
```
Ingestion Pipeline → PostgreSQL → Backend API → Frontend
                  ↓
                  Redis (cache)
                  ↓
                  OpenSearch (search)
                  ↓
                  AWS Bedrock (embeddings)
```

## How to Enable PostgreSQL Pipeline

### Step 1: Start Infrastructure (Docker)

**Install Docker Desktop:**
- Download: https://www.docker.com/products/docker-desktop/
- Install and start Docker Desktop

**Start Services:**
```bash
docker-compose up -d
```

This starts:
- PostgreSQL on port 5432
- Redis on port 6379
- OpenSearch on port 9200

**Verify Services:**
```bash
docker-compose 