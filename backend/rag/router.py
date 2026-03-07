"""
FastAPI router for RAG chat endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

# Global instances (initialized in main.py)
chat_service = None
vector_store = None


class ChatRequest(BaseModel):
    """Chat request model"""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    session_id: str = Field(default="default", description="Session ID for conversation tracking")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional search filters")


class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    in_scope: bool
    session_id: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    vector_store: Dict[str, Any]
    timestamp: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG-powered chat endpoint with conversational memory.
    
    Features:
    - Retrieves relevant context from vector store
    - Generates contextual responses
    - Maintains conversation history
    - Rejects out-of-scope queries
    
    Example:
    ```json
    {
        "query": "I need a free machine learning model for text classification",
        "session_id": "user123",
        "filters": {"resource_type": ["Model"], "pricing_type": ["free"]}
    }
    ```
    """
    try:
        if chat_service is None:
            raise HTTPException(
                status_code=503,
                detail="Chat service not initialized. Please check server configuration."
            )
        
        response = chat_service.chat(
            query=request.query,
            session_id=request.session_id,
            filters=request.filters
        )
        
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat request failed: {str(e)}"
        )


@router.delete("/chat/{session_id}")
async def clear_conversation(session_id: str):
    """
    Clear conversation history for a session.
    
    Args:
        session_id: Session identifier
    """
    try:
        if chat_service is None:
            raise HTTPException(status_code=503, detail="Chat service not initialized")
        
        chat_service.clear_conversation(session_id)
        
        return {
            "message": f"Conversation history cleared for session: {session_id}",
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Clear conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check for RAG system.
    
    Returns:
    - Vector store status
    - Index existence
    - Document count
    """
    try:
        if vector_store is None:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        health = vector_store.health_check()
        
        from datetime import datetime
        return HealthResponse(
            status=health['status'],
            vector_store=health,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def initialize_rag_services(bedrock_client, opensearch_client):
    """
    Initialize RAG services (called from main.py).
    
    Args:
        bedrock_client: BedrockClient instance
        opensearch_client: OpenSearchClient instance
    """
    global chat_service, vector_store
    
    try:
        from rag.vector_store import VectorStore
        from rag.chat_service import ChatService
        
        # Initialize vector store
        vector_store = VectorStore(
            opensearch_client=opensearch_client,
            bedrock_client=bedrock_client
        )
        
        # Ensure index exists
        vector_store.ensure_index_exists()
        
        # Initialize chat service
        chat_service = ChatService(
            vector_store=vector_store,
            bedrock_client=bedrock_client,
            min_confidence=0.3
        )
        
        logger.info("✅ RAG services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG services: {e}")
        return False
