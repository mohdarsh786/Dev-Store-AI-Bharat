# Dev-Store Backend Tests

## Quick Connection Test

Test AWS connections (Bedrock + OpenSearch):
```bash
python test_connections_simple.py
```

## Full Test Suite

Run all tests:
```bash
pytest
```

Run specific test:
```bash
pytest test_opensearch_client.py
pytest test_bedrock.py
```

## Test Files

- `test_connections_simple.py` - Quick AWS connection validation
- `test_aws_connection.py` - Comprehensive AWS setup test
- `test_opensearch_client.py` - OpenSearch client tests
- `test_bedrock.py` - Bedrock integration tests
- `test_rag_system.py` - RAG engine tests
- `test_search_service.py` - Search service tests
- `test_database_client.py` - Database client tests
- `test_models.py` - Data model tests
- `test_ranking_features.py` - Ranking algorithm tests

## Setup Tests

- `wait_and_setup.py` - Auto-retry OpenSearch setup script
