"""Gateway API smoke tests without external service startup."""
from contextlib import asynccontextmanager
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/devstore")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("OPENSEARCH_HOST", "localhost")
os.environ.setdefault("S3_BUCKET_BOILERPLATE", "devstore-boilerplate-templates")
os.environ.setdefault("S3_BUCKET_CRAWLER_DATA", "devstore-crawler-data")

from api_gateway import app


@pytest.fixture
def client():
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def no_op_lifespan(_app):
        yield

    app.router.lifespan_context = no_op_lifespan
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.router.lifespan_context = original_lifespan


def test_health(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"


def test_search(client):
    response = client.post("/api/v1/search", json={
        "query": "machine learning API",
        "limit": 5,
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "machine learning API"
    assert "results" in payload


def test_intent_search(client):
    response = client.post("/api/v1/search/intent", json={
        "query": "I need a free NLP model",
        "limit": 5,
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "I need a free NLP model"
    assert "results" in payload
