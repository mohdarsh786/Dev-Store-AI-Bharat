"""Health and monitoring routes."""

from __future__ import annotations

from datetime import datetime
import time

from fastapi import APIRouter, Request

router = APIRouter()


async def _get_redis(req: Request):
    redis_client = getattr(req.app.state, "redis", None)
    if redis_client is not None:
        return redis_client

    from clients.redis_client import RedisClient

    redis_client = RedisClient()
    await redis_client.connect()
    req.app.state.redis = redis_client
    return redis_client


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/detailed")
async def detailed_health_check(req: Request):
    start_time = time.time()
    checks = {}

    try:
        redis_client = await _get_redis(req)
        checks["redis"] = await redis_client.ping()
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

    try:
        bedrock_client = getattr(req.app.state, "bedrock", None)
        if bedrock_client is None:
            from clients.bedrock import BedrockClient

            bedrock_client = BedrockClient()
        checks["bedrock"] = bedrock_client.health_check()
    except Exception as e:
        checks["bedrock"] = False
        checks["bedrock_error"] = str(e)

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
    redis_client = await _get_redis(req)
    cache_stats = await redis_client.get_cache_stats()
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cache": cache_stats,
        "uptime_seconds": time.time() - req.app.state.start_time if hasattr(req.app.state, "start_time") else 0
    }


@router.get("/cache/stats")
async def get_cache_stats(req: Request):
    redis_client = await _get_redis(req)
    stats = await redis_client.get_cache_stats()
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "stats": stats
    }


@router.post("/cache/flush")
async def flush_cache(req: Request):
    redis_client = await _get_redis(req)
    success = await redis_client.flush_all()
    return {
        "message": "Cache flushed successfully" if success else "Failed to flush cache",
        "success": success,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/ingestion/status/latest")
async def latest_ingestion_status(source: str | None = None):
    from ingestion.repository import IngestionRepository

    repository = IngestionRepository()
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "status": repository.latest_run_status(source),
    }
