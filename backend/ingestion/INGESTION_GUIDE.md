# Ingestion Pipeline Guide

## Overview

The Dev Store ingestion pipeline fetches ML models, datasets, and tools from multiple external sources using direct HTTP API calls. Data is automatically deduplicated, normalized, and stored in the database with vector embeddings for semantic search—powered exclusively by **Neon (Metadata)** and **Pinecone (Vectors)**.

---

## 🚀 Technical Stack

- **Metadata Store:** Neon Serverless PostgreSQL
- **Vector Store:** Pinecone Serverless (us-east-1)
- **Embeddings:** AWS Bedrock (Titan Text Embeddings v2)
- **Caching:** Redis (Atomic locks & embedding cache)
- **Orchestration:** Python 3.11 Custom Pipeline

---

## 📂 File Structure

```text
backend/ingestion/
├── fetchers/                       # Source-specific data harvesters
│   ├── huggingface_fetcher.py      # HuggingFace (Models + Datasets)
│   ├── openrouter_fetcher.py       # OpenRouter (LLM Specs + Pricing)
│   ├── github_fetcher.py           # GitHub (Repos + Dev Tools)
│   ├── kaggle_fetcher.py           # Kaggle (Structured Datasets)
├── repositories/                   # Data persistence logic
├── services/                       # Core pipeline logic
│   ├── chunking_service.py         # Text chunking for long descriptions
│   ├── embedding_service.py        # Bedrock embedding generator
│   └── ranking_service.py          # Unified score computation (rank_score)
├── stages/                         # Atomic pipeline stages
├── orchestrator_production.py      # Main production entry point
└── INGESTION_GUIDE.md              # This document
```

---

## 🔄 Pipeline Stages

The production orchestrator executes 10 sequential stages to ensure data integrity and search relevance:

1.  **Acquire Lock:** Redis distributed lock prevents parallel pipeline executions.
2.  **Fetch:** Parallel data retrieval from all enabled harvesters.
3.  **Normalize:** Conversion of raw JSON payloads to the **Canonical Resource Schema**.
4.  **Deduplicate:** Merging identical resources across sources (e.g., same model on OpenRouter & HuggingFace).
5.  **Upsert (Neon):** Transactional insert/update into PostgreSQL with change detection.
6.  **Embed (Bedrock):** Generation of high-dimensional vectors via Amazon Titan.
7.  **Upsert (Pinecone):** Bulk upserting vectors and metadata to the Pinecone index.
8.  **Refresh Rankings:** Computation of global popularity and freshness scores.
9.  **Cache Invalidation:** Clearing old search and ranking results in Redis.
10. **Release Lock:** Freeing the distributed lock for the next run.

---

## 🛠️ Quick Start

### 1. Configure Environment
Add these to your `backend/.env`:
```bash
# Vector Store
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=devstore-index

# Embeddings
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1

# Metadata
DATABASE_URL=postgresql://user:pass@host:5432/devstore
```

### 2. Run the Pipeline
```bash
cd backend
python ingestion/run_production.py
```

### 3. Check Status
```bash
# Monitor the ingestion_runs table in Neon
SELECT * FROM ingestion_runs ORDER BY started_at DESC LIMIT 1;
```

---

## 📊 Performance & Limits

- **Throughput:** ~10-15 resources/sec (Limited by Bedrock embedding concurrency).
- **Latency:** ~3-5 minutes for a full sync of 2,500+ resources.
- **Scale:** Optimized for the 100K+ resource tier using Pinecone serverless indexing.

**License:** Apache License 2.0
