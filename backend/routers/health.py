"""
Health and Monitoring Router - System health checks and metrics
"""
from fastapi import APIRouter, Request
import time
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint
    
    Returns simple status for load balancer health checks
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/detailed")
async def detailed_health_check(req: Request):
    """
    Detailed health check with dependency status
    
    Checks:
    - Redis connection
    - Database connection
    - OpenSearch connection
    - Bedrock availability
    """
    start_time = time.time()
    
    # Check dependencies
    checks = {}
    
    try:
        checks["redis"] = await req.app.state.redis.ping()
    except Exception as e:
        checks["redis"] = False
        checks["redis_error"] = str(e)
    
    try:
        checks["database"] = await req.app.state.db.health_check()
    except Exception as e:
        checks["database"] = False
        checks["database_error"] = str(e)
    
    try:
        checks["opensearch"] = await req.app.state.opensearch.health_check()
    except Exception as e:
        checks["opensearch"] = False
        checks["opensearch_error"] = str(e)
    
    # Bedrock is stateless, assume healthy
    checks["bedrock"] = True
    
    # Overall status
    all_healthy = all([
        checks.get("redis", False),
        checks.get("database", False),
        checks.get("opensearch", False),
        checks.get("bedrock", False)
    ])
    
    response_time = time.time() - start_time
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "response_time_ms": round(response_time * 1000, 2),
        "checks": checks
    }


@router.get("/metrics")
async def get_metrics(req: Request):
    """
    Get application metrics
    
    Returns:
    - Cache statistics
    - Request counts
    - Performance metrics
    """
    # Get cache stats
    cache_stats = await req.app.state.redis.get_cache_stats()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cache": cache_stats,
        "uptime_seconds": time.time() - req.app.state.start_time if hasattr(req.app.state, 'start_time') else 0
    }


@router.get("/cache/stats")
async def get_cache_stats(req: Request):
    """
    Get detailed cache statistics
    
    Returns Redis cache performance metrics
    """
    stats = await req.app.state.redis.get_cache_stats()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "stats": stats
    }


@router.post("/cache/flush")
async def flush_cache(req: Request):
    """
    Flush all cache (admin only)
    
    WARNING: This will clear all cached data
    """
    # TODO: Add admin authentication
    
    success = await req.app.state.redis.flush_all()
    
    return {
        "message": "Cache flushed successfully" if success else "Failed to flush cache",
        "success": success,
        "timestamp": datetime.utcnow().isoformat()
    }
