# DevStore RAG: Intelligent Developer Assistant

The DevStore RAG (Retrieval-Augmented Generation) system provides context-aware assistance for developer tool discovery. It leverages **AWS Bedrock** for reasoning and **Pinecone** for low-latency semantic retrieval.

---

## 🚀 Key Features

- **Multilingual Support:** Handles English, Hindi, and Hinglish queries (e.g., *"Bhai, check kro best auth APIs"*).
- **Dual-Layer Retrieval:** Combines structured SQL filters from Neon with semantic vectors from Pinecone.
- **Intent Discovery:** Automatically extracts developer intent, platform requirements, and pricing constraints.
- **Contextual Reasoning:** Uses Claude 3.5 Sonnet to explain *why* a resource is recommended.

---

## 🏗️ Architecture

```text
User Query (English/Hinglish)
       │
       ▼
Logic Layer (FastAPI)
       │
       ├─► Intent Extraction (Bedrock Nova Micro)
       │
       ├─► Hybrid Search
       │     ├─ Metadata Filter (Neon SQL)
       │     └─ Vector Search (Pinecone KNN)
       │
       ├─► Context Injection (Top K Results)
       │
       └─► Final Response (Bedrock Claude 3.5 Sonnet)
```

---

## 🛠️ Configuration

Configure the RAG engine in your `backend/.env`:

```env
# AI Models
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0

# Vector Store
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=devstore-index
```

---

## 📡 API Endpoints

### `POST /api/v1/rag/chat`
The primary endpoint for interacting with the AI assistant.

**Request:**
```json
{
  "query": "Bhai, best payment gateway batao for India",
  "history": [],
  "filters": {
    "pricing": "free"
  }
}
```

**Response:**
```json
{
  "answer": "Based on the Indian ecosystem, I recommend **Razorpay**...",
  "sources": [
    { "id": "razorpay-api", "name": "Razorpay", "score": 0.98 }
  ],
  "confidence": 0.96
}
```

---

## 🎯 Dual-Layer Ranking

The RAG engine uses a weighted scoring system:
1. **Semantic Score (60%)**: Relevance to query intent.
2. **Popularity Score (30%)**: Normalized stars and downloads.
3. **Freshness Score (10%)**: Days since last update.

---

## 🧪 Testing the RAG Engine

```bash
cd backend
pytest tests/test_rag_engine.py
```

**License:** MIT
