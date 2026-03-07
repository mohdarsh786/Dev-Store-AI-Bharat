"""Pydantic models for ingestion pipeline state and canonical resources."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from .domain import HealthStatus, PricingType, ResourceType


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


class IngestionSource(str, Enum):
    GITHUB = "github"
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"
    OPENROUTER = "openrouter"


class IngestionStage(str, Enum):
    FETCH = "fetch"
    NORMALIZE = "normalize"
    DEDUPE = "dedupe"
    UPSERT = "upsert"
    EMBED = "embed"
    INDEX = "index"
    RANK = "rank"
    INVALIDATE_CACHE = "invalidate_cache"


class IngestionStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    SKIPPED = "skipped"


class CanonicalResource(BaseModel):
    """Canonical resource document shared across all ingestion sources."""

    model_config = ConfigDict(use_enum_values=True)

    source: IngestionSource
    resource_type: ResourceType
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1)
    source_url: str = Field(..., pattern=r"^https?://")
    documentation_url: Optional[str] = Field(default=None, pattern=r"^https?://")
    pricing_type: PricingType
    github_stars: int = Field(default=0, ge=0)
    download_count: int = Field(default=0, ge=0)
    active_users: int = Field(default=0, ge=0)
    health_status: HealthStatus = HealthStatus.HEALTHY
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_updated_at: Optional[datetime] = None
    raw_payload: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("tags", "categories")
    @classmethod
    def normalize_string_lists(cls, value: List[str]) -> List[str]:
        cleaned: List[str] = []
        for item in value:
            stripped = item.strip()
            if stripped and stripped not in cleaned:
                cleaned.append(stripped)
        return cleaned

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unique_key(self) -> str:
        normalized_url = self.source_url.rstrip("/").lower()
        return f"{_enum_value(self.source)}:{normalized_url}"

    def embedding_text(self) -> str:
        parts = [self.name, self.description, " ".join(self.tags), _enum_value(self.resource_type)]
        return "\n".join(part for part in parts if part).strip()

    def embedding_hash(self) -> str:
        return hashlib.sha256(self.embedding_text().encode("utf-8")).hexdigest()

    def content_hash(self) -> str:
        canonical_payload = {
            "source": _enum_value(self.source),
            "resource_type": _enum_value(self.resource_type),
            "name": self.name,
            "description": self.description,
            "source_url": self.source_url,
            "documentation_url": self.documentation_url,
            "pricing_type": _enum_value(self.pricing_type),
            "github_stars": self.github_stars,
            "download_count": self.download_count,
            "active_users": self.active_users,
            "tags": self.tags,
            "categories": self.categories,
            "metadata": self.metadata,
            "source_updated_at": self.source_updated_at.isoformat() if self.source_updated_at else None,
        }
        encoded = json.dumps(canonical_payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class IngestionCounters(BaseModel):
    fetched_count: int = 0
    inserted_count: int = 0
    updated_count: int = 0
    unchanged_count: int = 0
    failed_count: int = 0
    embedded_count: int = 0
    indexed_count: int = 0


class IngestionRunRecord(BaseModel):
    id: Optional[UUID] = None
    run_id: UUID
    source: str
    status: IngestionStatus
    started_at: datetime
    finished_at: Optional[datetime] = None
    stage: Optional[IngestionStage] = None
    top_failure_reason: Optional[str] = None
    partial_completion: bool = False
    progress: Dict[str, Any] = Field(default_factory=dict)
    counters: IngestionCounters = Field(default_factory=IngestionCounters)
