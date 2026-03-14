"""
Opt-in integration smoke test for live AWS connections.
"""
from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("DEVSTORE_RUN_AWS_TESTS") != "1",
    reason="Live AWS connection test is opt-in",
)


def test_live_aws_connections():
    boto3 = pytest.importorskip("boto3")
    pytest.importorskip("opensearchpy")
    pytest.importorskip("requests_aws4auth")

    from config import settings

    required_settings = [
        settings.aws_access_key_id,
        settings.aws_secret_access_key,
        settings.aws_region,
    ]
    if not all(required_settings):
        pytest.skip("AWS credentials are not configured for integration testing")

    session = boto3.Session(
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )

    identity = session.client("sts").get_caller_identity()
    assert identity["Account"]
    assert identity["Arn"]

