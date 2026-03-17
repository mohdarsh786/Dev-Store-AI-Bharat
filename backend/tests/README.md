# Dev-Store Backend Test Suite

Comprehensive testing for the Dev-Store FastAPI logic layer, ensuring robust interactions with Neon, Pinecone, and AWS Bedrock.

---

## 🚀 Execution

Run all tests:
```bash
pytest
```

Run with coverage report:
```bash
pytest --cov=services --cov=rag --cov-report=term-missing
```

---

## 📂 Key Test Categories

- **`test_connections_simple.py`**: Fast validation of AWS, Neon, and Pinecone connectivity.
- **`test_search_service.py`**: Critical path testing for the Dual-Layer Search engine.
- **`test_rag_engine.py`**: Validation of AI assistant responses and context injection.
- **`test_pinecone_client.py`**: Tests for vector upserts and semantic queries.
- **`test_database_client.py`**: Neon PostgreSQL connection and migration tests.
- **`test_ranking_features.py`**: Verification of normalized scoring and trending logic.

---

## 🛠️ Testing Infrastructure

- **Mocking:** Intensive use of `unittest.mock` for Bedrock and Pinecone to reduce API costs during development.
- **Database:** Uses a test schema in Neon to prevent production data corruption.
- **RAG Validation:** Automated checks for multilingual query parsing (English/Hinglish).

**License:** MIT
