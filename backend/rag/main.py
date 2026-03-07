"""
FastAPI application with RAG chat endpoint
"""
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from rag.models import ChatRequest, ChatResponse
from rag.rag_engine import RAGEngine
from rag.vector_store import VectorStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
vector_store: VectorStore = None
rag_engine: RAGEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global vector_store, rag_engine
    
    try:
        # Load configuration
        from config import settings
        
        logger.info("Initializing RAG system...")
        
        # Initialize Vector Store
        vector_store = VectorStore(
            host=settings.opensearch_host,
            port=settings.opensearch_port,
            region=settings.aws_region,
            index_name=settings.opensearch_index_name,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            vector_dimension=1024
        )
        
        # Check if index exists
        if not vector_store.index_exists():
            logger.warning("OpenSearch index does not exist! Run ingestor.py first.")
        else:
            doc_count = vector_store.get_document_count()
            logger.info(f"OpenSearch index ready with {doc_count} documents")
        
        # Initialize RAG Engine
        rag_engine = RAGEngine(
            vector_store=vector_store,
            bedrock_region=settings.aws_region,
            model_id=settings.bedrock_model_id,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            confidence_threshold=0.3
        )
        
        logger.info("✅ RAG system initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down RAG system...")


# Create FastAPI app
app = FastAPI(
    title="DevStore RAG API",
    description="AI-powered conversational search for developer tools",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "DevStore RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/chat",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if vector_store is None or rag_engine is None:
            return {
                "status": "unhealthy",
                "message": "RAG system not initialized"
            }
        
        # Check OpenSearch connection
        doc_count = vector_store.get_document_count()
        
        return {
            "status": "healthy",
            "opensearch": {
                "connected": True,
                "index": vector_store.index_name,
                "document_count": doc_count
            },
            "rag_engine": {
                "model": rag_engine.model_id,
                "ready": True
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Conversational RAG endpoint
    
    This endpoint:
    1. Validates the query is about developer tools
    2. Retrieves relevant context using hybrid search
    3. Generates a response using Bedrock with conversation memory
    4. Returns answer with sources and confidence score
    
    Example request:
    ```json
    {
        "query": "I need a Python web framework for building APIs",
        "conversation_history": [],
        "filters": {"category": "api"},
        "max_results": 5
    }
    ```
    """
    try:
        if rag_engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG system not initialized"
            )
        
        # Validate query length
        if len(request.query) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query must be at least 3 characters"
            )
        
        if len(request.query) > 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query must be less than 500 characters"
            )
        
        # Process chat request
        logger.info(f"Processing chat request: {request.query[:100]}")
        response = rag_engine.chat(request)
        
        logger.info(f"Chat response generated (confidence={response.confidence})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        if vector_store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector store not initialized"
            )
        
        doc_count = vector_store.get_document_count()
        
        return {
            "total_documents": doc_count,
            "index_name": vector_store.index_name,
            "vector_dimension": vector_store.vector_dimension,
            "status": "operational"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "rag.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
