"""
Test script for RAG system
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_vector_store():
    """Test VectorStore initialization and index creation"""
    print("\n" + "="*60)
    print("TEST 1: Vector Store")
    print("="*60)
    
    try:
        from config import settings
        from rag.vector_store import VectorStore
        
        vector_store = VectorStore(
            host=settings.opensearch_host,
            port=settings.opensearch_port,
            region=settings.aws_region,
            index_name=se