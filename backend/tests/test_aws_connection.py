"""
Opt-in AWS integration tests.
"""
from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("DEVSTORE_RUN_AWS_TESTS") != "1",
    reason="Live AWS integration tests are opt-in",
)


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("boto3")
    pytest.importorskip("opensearchpy")

    from aws_client import AWSClient
    from config import settings

    required_settings = [
        settings.aws_region,
        settings.opensearch_host,
        settings.opensearch_index_name,
        settings.aws_access_key_id,
        settings.aws_secret_access_key,
    ]
    if not all(required_settings):
        pytest.skip("AWS integration settings are incomplete")

    return AWSClient(
        region_name=settings.aws_region,
        opensearch_host=settings.opensearch_host,
        opensearch_index=settings.opensearch_index_name,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )


def test_credentials(client):
    assert client.session is not None


def test_opensearch_connection(client):
    assert client.opensearch_client is not None


def test_bedrock_embedding(client):
    embedding = client.generate_embedding("machine learning dataset for sentiment analysis")
    assert len(embedding) == 1024


def test_claude_invocation(client):
    response = client.invoke_claude(
        messages=[{"role": "user", "content": "Say hello in five words."}],
        max_tokens=50,
        temperature=0.1,
    )
    assert response


def test_end_to_end(client):
    embedding = client.generate_embedding("python web framework")
    results = client.knn_search(query_vector=embedding, k=3)
    assert isinstance(results, list)

