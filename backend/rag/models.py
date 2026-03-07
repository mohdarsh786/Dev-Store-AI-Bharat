"""
Pydantic models for RAG system data validation
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class ResourceMetadata(BaseModel):
    """Metadata for a resource"""
    language: Optional[str] = None
    forks: Optional[int] = None
    watchers: Optional[int] = None
    open_issues: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    pushed_at: Optional[str] = None
    size: Optional[int] = None
    has_wiki: Optional[bool] = None
    has_pages: Optional[bool] = None
    dataset_id: Optional[str] = None
    last_modified: Optional[str] = None
    private: Optional[bool] = None
    gated: Optional[bool] = None
    ref: Optional[str] = None
    usability_rating: Optional[float] = None


class DeveloperResource(BaseModel):
    """Schema for developer resources (APIs, Models, Datasets)"""
    name: str = Field(..., description="Resource name")
    description: str = Field(..., description="Resource description")
    source: str = Field(..., description="Source platform (github, huggingface, kaggle)")
    source_url: str = Field(..., description="URL to the resource")
    author: str = Field(..., description="Author/organization")
    stars: int = Field(default=0, description="Number of stars/likes")
    downloads: int = Field(default=0, description="Number of downloads")
    license: Optional[str] = Field(default="Unknown", description="License type")
    tags: List[str] = Field(default_factory=list, description="Tags/keywords")
    version: Optional[str] = Field(default="latest", description="Version")
    category: str = Field(..., description="Category (api, dataset, model)")
    thumbnail_url: Optional[str] = None
    readme_url: Optional[str] = None
    metadata: Optional[ResourceMetadata] = None
    scraped_at: Optional[str] = None
    
    @validator('category')
    def validate_category(cls, v):
        """Ensure category is one of the allowed values"""
        allowed = ['api', 'dataset', 'model']
        if v.lower() not in allowed:
            # Try to infer from context
            return 'api'  # default
        return v.lower()
    
    @validator('description')
    def validate_description(cls, v):
        """Ensure description is not empty"""
        if not v or len(v.strip()) < 10:
            raise ValueError("Description must be at least 10 characters")
        return v.strip()
    
    @validator('tags', pre=True)
    def validate_tags(cls, v):
        """Clean and validate tags"""
        if isinstance(v, str):
            return [v]
        if not isinstance(v, list):
            return []
        # Remove empty tags and clean
        return [tag.strip() for tag in v if tag and isinstance(tag, str)]


class SearchQuery(BaseModel):
    """Search query model"""
    query: str = Field(..., min_length=3, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    limit: int = Field(default=10, ge=1, le=50)


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant', 'system']:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        return v


class ChatRequest(BaseModel):
    """Chat request with conversation history"""
    query: str = Field(..., min_length=3, description="User query")
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    max_results: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float = Field(ge=0.0, le=1.0)
    query: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
