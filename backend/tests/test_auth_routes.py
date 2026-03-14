"""Authentication route tests."""
from contextlib import asynccontextmanager
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/devstore")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("OPENSEARCH_HOST", "localhost")
os.environ.setdefault("S3_BUCKET_BOILERPLATE", "devstore-boilerplate-templates")
os.environ.setdefault("S3_BUCKET_CRAWLER_DATA", "devstore-crawler-data")

from api_gateway import app
from config import settings


@pytest.fixture
def client():
    original_lifespan = app.router.lifespan_context
    original_secret = settings.secret_key

    @asynccontextmanager
    async def no_op_lifespan(_app):
        yield

    app.router.lifespan_context = no_op_lifespan
    settings.secret_key = "test-secret-key"
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.router.lifespan_context = original_lifespan
        settings.secret_key = original_secret


def test_token_exchange_and_me(client):
    user_row = {
        "id": "5d3edbf2-4d37-4894-a2ff-4d22694797d0",
        "cognito_id": "cognito-user-123",
        "email": "user@example.com",
        "preferred_language": "en",
        "tech_stack": ["python", "fastapi"],
        "preferences": {"theme": "dark"},
        "created_at": None,
        "last_login": None,
    }

    with patch("services.auth.DatabaseClient") as _:
        with patch("services.auth.AuthService._upsert_user", return_value=user_row):
            token_response = client.post(
                "/api/v1/auth/token",
                json={
                    "cognito_id": user_row["cognito_id"],
                    "email": user_row["email"],
                    "tech_stack": user_row["tech_stack"],
                    "preferences": user_row["preferences"],
                },
            )

    assert token_response.status_code == 200
    token_payload = token_response.json()
    assert token_payload["token_type"] == "bearer"
    assert token_payload["user"]["email"] == user_row["email"]

    with patch("services.auth.DatabaseClient") as _:
        with patch(
            "services.auth.AuthService.get_current_user",
            return_value={
                "id": user_row["id"],
                "cognito_id": user_row["cognito_id"],
                "email": user_row["email"],
                "preferred_language": "en",
                "tech_stack": user_row["tech_stack"],
                "preferences": user_row["preferences"],
                "created_at": None,
                "last_login": None,
            },
        ):
            me_response = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token_payload['access_token']}"},
            )

    assert me_response.status_code == 200
    assert me_response.json()["email"] == user_row["email"]
