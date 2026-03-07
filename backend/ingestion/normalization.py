"""Canonical normalization helpers for ingestion sources."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from models import CanonicalResource, HealthStatus, IngestionSource, PricingType, ResourceType


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _normalize_type(category: str) -> ResourceType:
    lowered = (category or "").strip().lower()
    mapping = {
        "api": ResourceType.API,
        "model": ResourceType.MODEL,
        "dataset": ResourceType.DATASET,
    }
    return mapping.get(lowered, ResourceType.API)


def _normalize_pricing(source: str, payload: Dict[str, Any]) -> PricingType:
    pricing = payload.get("metadata", {}).get("pricing_type")
    if pricing == "paid":
        return PricingType.PAID
    if pricing == "freemium":
        return PricingType.FREEMIUM
    if source == IngestionSource.OPENROUTER.value and payload.get("metadata", {}).get("pricing", {}):
        price_prompt = float(payload["metadata"]["pricing"].get("prompt", 0) or 0)
        price_completion = float(payload["metadata"]["pricing"].get("completion", 0) or 0)
        return PricingType.FREE if price_prompt == 0 and price_completion == 0 else PricingType.PAID
    return PricingType.FREE


def _base_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "author": payload.get("author"),
        "license": payload.get("license"),
        "version": payload.get("version"),
        "thumbnail_url": payload.get("thumbnail_url"),
        "readme_url": payload.get("readme_url"),
        "legacy_source_category": payload.get("category"),
    }


def canonicalize_resource(
    source: IngestionSource,
    payload: Dict[str, Any],
    raw_payload: Dict[str, Any],
) -> CanonicalResource:
    """Convert legacy fetcher output into the canonical ingestion schema."""
    metadata = _base_metadata(payload)
    resource_type = _normalize_type(payload.get("category", "api"))

    if resource_type == ResourceType.API:
        metadata["api"] = {
            "language": payload.get("metadata", {}).get("language"),
            "forks": payload.get("metadata", {}).get("forks", 0),
            "watchers": payload.get("metadata", {}).get("watchers", 0),
            "open_issues": payload.get("metadata", {}).get("open_issues", 0),
            "created_at": payload.get("metadata", {}).get("created_at"),
            "updated_at": payload.get("metadata", {}).get("updated_at"),
            "pushed_at": payload.get("metadata", {}).get("pushed_at"),
            "size": payload.get("metadata", {}).get("size", 0),
            "has_wiki": payload.get("metadata", {}).get("has_wiki", False),
            "has_pages": payload.get("metadata", {}).get("has_pages", False),
        }
    elif resource_type == ResourceType.MODEL:
        metadata["model"] = {
            "model_id": payload.get("metadata", {}).get("model_id"),
            "pipeline_tag": payload.get("metadata", {}).get("pipeline_tag"),
            "library_name": payload.get("metadata", {}).get("library_name"),
            "context_length": payload.get("metadata", {}).get("context_length", 0),
            "pricing": payload.get("metadata", {}).get("pricing", {}),
            "architecture": payload.get("metadata", {}).get("architecture", {}),
            "top_provider": payload.get("metadata", {}).get("top_provider", {}),
            "per_request_limits": payload.get("metadata", {}).get("per_request_limits"),
            "private": payload.get("metadata", {}).get("private", False),
            "gated": payload.get("metadata", {}).get("gated", False),
        }
    else:
        metadata["dataset"] = {
            "dataset_id": payload.get("metadata", {}).get("dataset_id"),
            "ref": payload.get("metadata", {}).get("ref"),
            "last_updated": payload.get("metadata", {}).get("last_updated"),
            "usability_rating": payload.get("metadata", {}).get("usability_rating", 0),
            "private": payload.get("metadata", {}).get("private", False),
            "gated": payload.get("metadata", {}).get("gated", False),
        }

    source_updated_at = (
        _parse_datetime(payload.get("metadata", {}).get("pushed_at"))
        or _parse_datetime(payload.get("metadata", {}).get("updated_at"))
        or _parse_datetime(payload.get("metadata", {}).get("last_modified"))
        or _parse_datetime(payload.get("metadata", {}).get("lastUpdated"))
        or _parse_datetime(payload.get("metadata", {}).get("created_at"))
    )

    documentation_url = payload.get("readme_url") or payload.get("source_url")
    description = (payload.get("description") or "").strip()
    if not description:
        description = f"{source.value.title()} resource: {payload.get('name', 'unknown')}"

    categories = [payload.get("category", resource_type.value)]
    if pipeline_tag := payload.get("metadata", {}).get("pipeline_tag"):
        categories.append(pipeline_tag.replace("-", " "))

    return CanonicalResource(
        source=source,
        resource_type=resource_type,
        name=(payload.get("name") or "").strip(),
        description=description,
        source_url=payload.get("source_url"),
        documentation_url=documentation_url,
        pricing_type=_normalize_pricing(source.value, payload),
        github_stars=payload.get("stars", 0) or 0,
        download_count=payload.get("downloads", 0) or 0,
        active_users=payload.get("metadata", {}).get("watchers", 0) or 0,
        health_status=HealthStatus.HEALTHY,
        tags=payload.get("tags", []),
        categories=categories,
        metadata=metadata,
        source_updated_at=source_updated_at,
        raw_payload=raw_payload,
    )
