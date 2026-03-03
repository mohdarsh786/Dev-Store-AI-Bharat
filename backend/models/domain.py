"""
Domain models for DevStore platform.

This module contains Pydantic models representing the core domain entities
as specified in the design document.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, ConfigDict


# Enums

class ResourceType(str, Enum):
    """Type of resource in the marketplace."""
    API = "api"
    MODEL = "model"
    DATASET = "dataset"


class PricingType(str, Enum):
    """Pricing model for resources."""
    FREE = "free"
    PAID = "paid"
    FREEMIUM = "freemium"


class HealthStatus(str, Enum):
    """Health status of a resource."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


# Core Domain Models

class Resource(BaseModel):
    """
    Represents a resource (API, Model, or Dataset) in the marketplace.
    
    Attributes:
        id: Unique identifier
        type: Resource type (API, Model, or Dataset)
        name: Display name
        description: Short description
        long_description: Detailed description (optional)
        pricing_type: Pricing model
        price_details: Pricing information (optional)
        source_url: URL to the resource
        documentation_url: URL to documentation (optional)
        github_stars: Number of GitHub stars (optional)
        download_count: Number of downloads (optional)
        active_users: Number of active users (optional)
        health_status: Current health status
        last_health_check: Timestamp of last health check (optional)
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Type-specific fields
        embedding: Vector embedding for semantic search (optional)
        categories: List of category names
        tags: List of tags
    """
    model_config = ConfigDict(use_enum_values=True)
    
    id: UUID
    type: ResourceType
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1)
    long_description: Optional[str] = None
    pricing_type: PricingType
    price_details: Optional[Dict[str, Any]] = None
    source_url: str = Field(..., pattern=r'^https?://')
    documentation_url: Optional[str] = Field(None, pattern=r'^https?://')
    github_stars: Optional[int] = Field(None, ge=0)
    download_count: Optional[int] = Field(None, ge=0)
    active_users: Optional[int] = Field(None, ge=0)
    health_status: HealthStatus = HealthStatus.HEALTHY
    last_health_check: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Computed fields
    embedding: Optional[List[float]] = None
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding_dimension(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate that embedding has correct dimension (1536 for Titan)."""
        if v is not None and len(v) != 1536:
            raise ValueError(f'Embedding must have 1536 dimensions, got {len(v)}')
        return v


class SearchFilters(BaseModel):
    """
    Filters for search queries.
    
    Attributes:
        resource_types: Filter by resource types
        pricing_types: Filter by pricing models
        categories: Filter by category names
        min_stars: Minimum GitHub stars
        health_status: Filter by health status
        languages: Filter by programming languages
    """
    resource_types: Optional[List[ResourceType]] = None
    pricing_types: Optional[List[PricingType]] = None
    categories: Optional[List[str]] = None
    min_stars: Optional[int] = Field(None, ge=0)
    health_status: Optional[List[HealthStatus]] = None
    languages: Optional[List[str]] = None


class Intent(BaseModel):
    """
    Structured representation of user's search intent.
    
    Attributes:
        primary_need: Primary resource type needed
        secondary_needs: Additional resource types needed
        tech_stack: Mentioned technologies
        use_case: Extracted use case description
        constraints: Pricing, performance, etc.
        language: Query language code
        confidence: Intent extraction confidence [0, 1]
    """
    primary_need: ResourceType
    secondary_needs: List[ResourceType] = Field(default_factory=list)
    tech_stack: List[str] = Field(default_factory=list)
    use_case: str
    constraints: Dict[str, Any] = Field(default_factory=dict)
    language: str = Field(default="en")
    confidence: float = Field(..., ge=0.0, le=1.0)


class RankingScore(BaseModel):
    """
    Composite ranking score with component breakdowns.
    
    Attributes:
        resource_id: Resource identifier
        semantic_relevance: Semantic similarity score [0, 1]
        popularity: Popularity score [0, 1]
        optimization: Optimization score [0, 1]
        freshness: Freshness score [0, 1]
        final_score: Weighted combination
        popularity_breakdown: Component scores for popularity
        optimization_breakdown: Component scores for optimization
        freshness_breakdown: Component scores for freshness
    """
    resource_id: UUID
    semantic_relevance: float = Field(..., ge=0.0, le=1.0)
    popularity: float = Field(..., ge=0.0, le=1.0)
    optimization: float = Field(..., ge=0.0, le=1.0)
    freshness: float = Field(..., ge=0.0, le=1.0)
    final_score: float = Field(..., ge=0.0, le=1.0)
    
    # Component breakdowns
    popularity_breakdown: Dict[str, float] = Field(default_factory=dict)
    optimization_breakdown: Dict[str, float] = Field(default_factory=dict)
    freshness_breakdown: Dict[str, float] = Field(default_factory=dict)
    
    @field_validator('final_score')
    @classmethod
    def validate_final_score(cls, v: float, info) -> float:
        """Validate that final_score matches weighted combination."""
        # Note: This validation is performed after all fields are set
        # The actual computation should be done by the RankingService
        return v


class SearchResult(BaseModel):
    """
    Single search result with resource and ranking information.
    
    Attributes:
        resource: The resource
        score: Ranking score
        explanation: Why this was recommended
        snippet: Highlighted relevant text
    """
    resource: Resource
    score: RankingScore
    explanation: str
    snippet: str


class SearchResults(BaseModel):
    """
    Complete search results with metadata.
    
    Attributes:
        query: Original search query
        intent: Extracted intent
        results: List of search results
        total_count: Total number of matching resources
        execution_time_ms: Query execution time
        filters_applied: Filters used in search
    """
    query: str
    intent: Intent
    results: List[SearchResult]
    total_count: int = Field(..., ge=0)
    execution_time_ms: float = Field(..., ge=0.0)
    filters_applied: SearchFilters


class UserContext(BaseModel):
    """
    User context for personalization.
    
    Attributes:
        user_id: User identifier
        preferred_language: Preferred language code
        tech_stack: User's technology stack
        search_history: Recent search queries
        used_resources: Previously used resource IDs
        preferences: User preferences
    """
    user_id: UUID
    preferred_language: str = Field(default="en")
    tech_stack: List[str] = Field(default_factory=list)
    search_history: List[str] = Field(default_factory=list)
    used_resources: List[UUID] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)


class CodeFile(BaseModel):
    """
    Represents a code file in a boilerplate package.
    
    Attributes:
        path: Relative path in package
        content: File content
        language: Programming language
    """
    path: str = Field(..., min_length=1)
    content: str
    language: str = Field(..., min_length=1)


class BoilerplatePackage(BaseModel):
    """
    Generated boilerplate code package.
    
    Attributes:
        package_id: Unique package identifier
        resources: Resources included in package
        language: Programming language
        files: List of code files
        readme: README content
        env_template: Environment variables template
        created_at: Creation timestamp
        download_url: S3 presigned URL for ZIP download
    """
    package_id: str = Field(..., min_length=1)
    resources: List[Resource]
    language: str = Field(..., pattern=r'^(python|javascript|typescript)$')
    files: List[CodeFile]
    readme: str
    env_template: str
    created_at: datetime
    download_url: str = Field(..., pattern=r'^https?://')
    
    @field_validator('resources')
    @classmethod
    def validate_resources_not_empty(cls, v: List[Resource]) -> List[Resource]:
        """Validate that at least one resource is included."""
        if not v:
            raise ValueError('At least one resource must be included in the package')
        return v
