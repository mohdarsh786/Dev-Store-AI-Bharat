"""
Basic smoke tests for RAG modules.
"""
from __future__ import annotations

import pytest


def test_rag_modules_importable():
    pytest.importorskip("opensearchpy")
    pytest.importorskip("pydantic_settings")

    from rag import models, router, vector_store  # noqa: F401

