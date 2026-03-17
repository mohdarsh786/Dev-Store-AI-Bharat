from dotenv import load_dotenv
load_dotenv()
from dotenv import load_dotenv
load_dotenv()
from dotenv import load_dotenv
load_dotenv()
"""
DevStore Backend - Main FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from routers import auth, health, search, resources
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DevStore API",
    description="AI-Powered Developer Marketplace API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router)
app.include_router(resources.router)
app.include_router(auth.router)
app.include_router(health.router)
try:
    from routers import rag
    app.include_router(rag.router)
except ImportError as e:
    logger.warning(f"RAG router not available: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize Serverless clients"""
    try:
        from clients.bedrock import BedrockClient
        bedrock_client = BedrockClient()
        logger.info("✅ Bedrock interface initialized")
    except Exception as e:
        logger.warning(f"⚠️ Initializing services error: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "DevStore API", "version": "1.0.0"}


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "devstore-api"
    }


# Lambda handler
handler = Mangum(app)
