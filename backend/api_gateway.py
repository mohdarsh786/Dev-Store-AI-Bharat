"""
Centralized API Gateway for DevStore
Consolidates all critical endpoints for EC2 deployment with Redis caching
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging

from routers import search, resources, categories, boilerplate, users, health
from clients.redis_client import RedisClient
from clients.database import DatabaseClient
from clients.opensearch import OpenSearchClient
from clients.bedrock import BedrockClient
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _is_healthy(result) -> bool:
    """Normalize boolean and dict-based health checks."""
    if isinstance(result, dict):
        return result.get("status") == "healthy"
    return bool(result)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown events"""
    # Startup
    logger.info("Starting DevStore API Gateway...")
    app.state.start_time = time.time()
    
    # Initialize clients
    app.state.redis = RedisClient()
    await app.state.redis.connect()
    
    app.state.db = DatabaseClient()
    
    app.state.opensearch = OpenSearchClient()
    
    app.state.bedrock = BedrockClient()
    
    logger.info("All clients initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down DevStore API Gateway...")
    await app.state.redis.disconnect()
    app.state.db.close()
    app.state.opensearch.close()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="DevStore Centralized API",
    description="AI-Powered Developer Marketplace - Centralized API Gateway",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing"""
    start_time = time.time()
    
    # Add request ID for tracing
    request_id = request.headers.get("X-Request-ID", f"req-{int(time.time() * 1000)}")
    
    logger.info(f"Request started: {request.method} {request.url.path} [ID: {request_id}]")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"[ID: {request_id}] [Status: {response.status_code}] "
            f"[Time: {process_time:.3f}s]"
        )
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"[ID: {request_id}] [Error: {str(e)}] [Time: {process_time:.3f}s]"
        )
        raise


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "details": str(exc) if settings.environment == "development" else None
        }
    )


# Include routers.
# `search.router` and `resources.router` already declare `/api/v1`,
# so adding another prefix here would publish them at `/api/v1/api/v1/...`.
app.include_router(search.router, tags=["Search"])
app.include_router(resources.router, tags=["Resources"])
app.include_router(categories.router, prefix="/api/v1", tags=["Categories"])
app.include_router(boilerplate.router, prefix="/api/v1", tags=["Boilerplate"])
app.include_router(users.router, prefix="/api/v1", tags=["Users"])
app.include_router(health.router, prefix="/api/v1", tags=["Health"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "DevStore Centralized API",
        "version": "2.0.0",
        "deployment": "EC2 with Redis",
        "docs": "/api/docs"
    }


@app.get("/api/v1/status")
async def api_status(request: Request):
    """API status with dependency health checks"""
    redis_status = await request.app.state.redis.ping()
    db_status = request.app.state.db.health_check()
    opensearch_status = request.app.state.opensearch.health_check()
    
    all_healthy = all([
        _is_healthy(redis_status),
        _is_healthy(db_status),
        _is_healthy(opensearch_status),
    ])
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": time.time(),
        "dependencies": {
            "redis": "healthy" if _is_healthy(redis_status) else "unhealthy",
            "database": "healthy" if _is_healthy(db_status) else "unhealthy",
            "opensearch": "healthy" if _is_healthy(opensearch_status) else "unhealthy",
            "bedrock": "healthy"  # Bedrock is stateless
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )
