# Dev-Store RAG System - Production Guide

## Overview

This is a production-ready RAG (Retrieval-Augmented Generation) system for Dev-Store that provides:

- **Automated Ingestion**: Reads JSON datasets, generates embeddings, and indexes in OpenSearch
- **Hybrid Search**: Combines vector semantic search (70%) + keyword BM25 (30%) for optimal ranking
- **Conversational Chat**: `/chat` endpoint with memory and context-aware responses
- **Out-of-Scope Rejection**: Automatically rejects non-developer-tool queries
- **Error Handling**: Graceful handling of empty results, Bedrock throttling, and OpenSearch timeouts

## Architecture

```
┌─────────────────┐
│  JSON Files     │
│  (GitHub, HF,   │
│   Kaggle, etc)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Ingestor  │
│  - Validation   │
│  - Chunking     │
│  - Embeddings   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OpenSearch     │
│  - k-NN Index   │
│  - BM25 Index   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │
│  - Hybrid Search│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chat Service   │
│  - RAG Logic    │
│  - Memory       │
│  - LLM (Claude) │
└─────────────────┘
```

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Ensure your `.env` file has:

```env
AWS_REGION=ap-northeast-3
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

OPENSEARCH_HOST=your-collection.ap-northeast-3.aoss.amazonaws.com
OPENSEARCH_PORT=443
OPENSEARCH_USE_SSL=true
OPENSEARCH_INDEX_NAME=devstore_resources

BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
```

### 3. Run Data Ingestion

```bash
python run_rag_ingestion.py
```

This will:
- Load all JSON files from `backend/` directory
- Validate each resource with Pydantic
- Generate embeddings using Bedrock Titan v2 (1024 dimensions)
- Index documents in OpenSearch with k-NN support

**Expected Output:**
```
============================================================
RAG DATA INGESTION PIPELINE
============================================================
Initializing clients...
✅ OpenSearch connected: your-cluster
✅ Index already exists
Initializing data ingestor...

Starting ingestion...
------------------------------------------------------------
Loaded 100 records from backend/github_resources.json
...

============================================================
INGESTION COMPLETE
============================================================
Files processed: 4
Total records: 50000
Successfully indexed: 49850
Failed: 150
Success rate: 99.7%
============================================================
```

## API Endpoints

### POST `/api/v1/rag/chat`

RAG-powered conversational chat with memory.

**Request:**
```json
{
  "query": "I need a free machine learning model for text classification",
  "session_id": "user123",
  "filters": {
    "resource_type": ["Model"],
    "pricing_type": ["free"]
  }
}
```

**Response:**
```json
{
  "answer": "Based on the search results, here are some excellent free ML models for text classification:\n\n1. **DistilBERT** - A lightweight BERT model that's 60% faster while retaining 97% of BERT's performance...",
  "sources": [
    {
      "name": "distilbert-base-uncased",
      "resource_type": "Model",
      "description": "...",
      "source_url": "https://huggingface.co/...",
      "github_stars": 5000,
      "downloads": 1000000
    }
  ],
  "confidence": 0.85,
  "in_scope": true,
  "session_id": "user123",
  "timestamp": "2026-03-07T10:30:00Z"
}
```

### DELETE `/api/v1/rag/chat/{session_id}`

Clear conversation history for a session.

**Response:**
```json
{
  "message": "Conversation history cleared for session: user123",
  "session_id": "user123"
}
```

### GET `/api/v1/rag/health`

Health check for RAG system.

**Response:**
```json
{
  "status": "healthy",
  "vector_store": {
    "status": "healthy",
    "opensearch": {
      "status": "healthy",
      "cluster_name": "your-cluster",
      "response_time_ms": 45.2
    },
    "index_exists": true,
    "document_count": 49850,
    "index_name": "devstore_resources"
  },
  "timestamp": "2026-03-07T10:30:00Z"
}
```

## Key Features

### 1. Hybrid Search

Combines two search approaches:
- **Vector Search (70%)**: Semantic similarity using k-NN on embeddings
- **Keyword Search (30%)**: BM25 for exact term matching

```python
# In vector_store.py
results = vector_store.hybrid_search(
    query="python web framework",
    alpha=0.7,  # 70% vector, 30% keyword
    k=10
)
```

### 2. Conversational Memory

Maintains conversation history per session:

```python
# Automatic memory management
chat_service.chat(
    query="Tell me more about the first one",
    session_id="user123"
)
# Remembers previous context
```

### 3. Out-of-Scope Detection

Automatically rejects non-developer queries:

```python
# Query: "What's the weather today?"
# Response: "I'm sorry, but I can only help with questions about developer tools..."
```

### 4. Confidence Filtering

Only returns high-confidence responses:

```python
if confidence < 0.3:
    return "I couldn't find a matching resource with high confidence..."
```

## Module Structure

```
backend/rag/
├── __init__.py           # Module exports
├── ingestor.py           # Data ingestion pipeline
├── vector_store.py       # Hybrid search implementation
├── chat_service.py       # RAG chat with memory
└── router.py             # FastAPI endpoints
```

### ingestor.py

- **ResourceSchema**: Pydantic model for validation
- **DataIngestor**: Handles JSON loading, validation, embedding generation, and indexing
- **Features**: Chunking, batch processing, error handling

### vector_store.py

- **VectorStore**: Manages OpenSearch operations
- **hybrid_search()**: Combines vector + keyword search
- **Features**: Health checks, connection management, query optimization

### chat_service.py

- **ChatService**: RAG-powered chat
- **ConversationMemory**: Session-based history
- **Features**: Context retrieval, response generation, scope detection

### router.py

- FastAPI endpoints for chat, health, and session management
- Request/response models with Pydantic
- Error handling and logging

## Error Handling

### Empty Search Results

```python
if not results:
    return {
        "answer": "I couldn't find a matching resource in our database. Could you try rephrasing?",
        "sources": [],
        "confidence": 0.0
    }
```

### Bedrock Throttling

```python
# Automatic retry with exponential backoff
for attempt in range(max_retries):
    try:
        response = bedrock.generate_embedding(text)
        break
    except ClientError as e:
        if e.response['Error']['Code'] == 'ThrottlingException':
            delay = base_delay * (2 ** attempt)
            time.sleep(min(delay, max_delay))
```

### OpenSearch Timeouts

```python
try:
    results = opensearch.search(query)
except TransportError as e:
    logger.error(f"OpenSearch timeout: {e}")
    return []  # Graceful degradation
```

## Testing

### Test Ingestion

```bash
python run_rag_ingestion.py
```

### Test Chat Endpoint

```bash
curl -X POST http://localhost:8000/api/v1/rag/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I need a Python web framework",
    "session_id": "test123"
  }'
```

### Test Health Check

```bash
curl http://localhost:8000/api/v1/rag/health
```

## Performance Optimization

### Batch Processing

```python
# Process documents in batches of 10
ingestor = DataIngestor(batch_size=10)
```

### Embedding Caching

```python
# Cache embeddings to avoid regeneration
self._embedding_cache[text] = embedding
```

### Connection Pooling

```python
# Reuse OpenSearch connections
opensearch_client = OpenSearchClient()  # Singleton pattern
```

## Monitoring

### Logs

```python
# All operations are logged
logger.info(f"Hybrid search returned {len(results)} results")
logger.error(f"Embedding generation failed: {e}")
```

### Metrics

- Document count: `GET /api/v1/rag/health`
- Response times: Logged in health check
- Success rates: Tracked during ingestion

## Troubleshooting

### Issue: No results returned

**Solution**: Check if index has documents
```bash
curl http://localhost:8000/api/v1/rag/health
# Check "document_count" field
```

### Issue: Bedrock throttling

**Solution**: Reduce batch size or add delays
```python
ingestor = DataIngestor(batch_size=5)  # Reduce from 10
```

### Issue: OpenSearch connection timeout

**Solution**: Check network and credentials
```python
health = opensearch_client.health_check()
print(health)
```

## Production Checklist

- [ ] Environment variables configured
- [ ] OpenSearch index created with k-NN support
- [ ] Data ingestion completed successfully
- [ ] Health check returns "healthy"
- [ ] Chat endpoint tested with sample queries
- [ ] Logging configured for production
- [ ] Error monitoring set up
- [ ] Rate limiting configured (if needed)

## Next Steps

1. **Scale Ingestion**: Add more data sources
2. **Improve Ranking**: Tune alpha parameter for hybrid search
3. **Add Analytics**: Track popular queries and response quality
4. **Optimize Performance**: Add caching layer (Redis)
5. **Enhanced Memory**: Persist conversation history to database

## Support

For issues or questions:
- Check logs: `tail -f backend/logs/app.log`
- Review health endpoint: `GET /api/v1/rag/health`
- Test individual components: `python -m pytest backend/tests/`
