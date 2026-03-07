"""
RAG (Retrieval-Augmented Generation) module for Dev-Store.

This module provides production-ready RAG capabilities including:
- Data ingestion from JSON files
- Hybrid search (Vector + BM25)
- Conversational chat with memory
- Out-of-scope query rejection
"""

from rag.ingestor import DataIngestor, ResourceSchema
from rag.vector_store import VectorStore
from rag.chat_service import ChatService, ConversationMemory
from rag.router import router, initialize_rag_services

__all__ = [
    'DataIngestor',
    'ResourceSchema',
    'VectorStore',
    'ChatService',
    'ConversationMemory',
    'router',
    'initialize_rag_services'
]
