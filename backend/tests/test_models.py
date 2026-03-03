"""
Unit tests for Pydantic domain models.

Tests basic validation rules and model instantiation.
"""

import pytest
from datetime import datetime
from uuid import uuid4
from pydantic import ValidationError

from models import (
    Resource,
    ResourceType,
    PricingType,
    HealthStatus,
    SearchFilters,
    Intent,
    RankingScore,
    SearchResult,
    SearchResults,
    UserContext,
    CodeFile,
    BoilerplatePackage,
)


class TestResourceModel:
    """Tests for Resource model."""
    
    def test_resource_creation_with_valid_data(self):
        """Test creating a resource with all required fields."""
        resource = Resource(
            id=uuid4(),
            type=ResourceType.API,
            name="Test API",
            description="A test API for testing",
            pricing_type=PricingType.FREE,
            source_url="https://example.com/api",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        assert resource.type == ResourceType.API
        assert resource.name == "Test API"
        assert resource.pricing_type == PricingType.FREE
        assert resource.health_status == HealthStatus.HEALTHY
    
    def test_resource_with_invalid_url(self):
        """Test that invalid URLs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Resource(
                id=uuid4(),
                type=ResourceType.API,
                name="Test API",
                description="A test API",
                pricing_type=PricingType.FREE,
                source_url="not-a-url",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        
        assert "source_url" in str(exc_info.value)
    
    def test_resource_with_negative_stars(self):
        """Test that negative GitHub stars are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Resource(
                id=uuid4(),
                type=ResourceType.API,
                name="Test API",
                description="A test API",
                pricing_type=PricingType.FREE,
                source_url="https://example.com",
                github_stars=-10,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        
        assert "github_stars" in str(exc_info.value)
    
    def test_resource_with_invalid_embedding_dimension(self):
        """Test that embeddings with wrong dimension are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Resource(
                id=uuid4(),
                type=ResourceType.API,
                name="Test API",
                description="A test API",
                pricing_type=PricingType.FREE,
                source_url="https://example.com",
                embedding=[0.1] * 100,  # Wrong dimension
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        
        assert "1536 dimensions" in str(exc_info.value)
    
    def test_resource_with_valid_embedding(self):
        """Test that embeddings with correct dimension are accepted."""
        resource = Resource(
            id=uuid4(),
            type=ResourceType.API,
            name="Test API",
            description="A test API",
            pricing_type=PricingType.FREE,
            source_url="https://example.com",
            embedding=[0.1] * 1536,  # Correct dimension
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        assert len(resource.embedding) == 1536


class TestSearchFilters:
    """Tests for SearchFilters model."""
    
    def test_empty_filters(self):
        """Test creating filters with no values."""
        filters = SearchFilters()
        
        assert filters.resource_types is None
        assert filters.pricing_types is None
        assert filters.categories is None
    
    def test_filters_with_values(self):
        """Test creating filters with specific values."""
        filters = SearchFilters(
            resource_types=[ResourceType.API, ResourceType.MODEL],
            pricing_types=[PricingType.FREE],
            min_stars=100,
        )
        
        assert len(filters.resource_types) == 2
        assert filters.pricing_types == [PricingType.FREE]
        assert filters.min_stars == 100


class TestIntent:
    """Tests for Intent model."""
    
    def test_intent_creation(self):
        """Test creating an intent with valid data."""
        intent = Intent(
            primary_need=ResourceType.API,
            secondary_needs=[ResourceType.MODEL],
            tech_stack=["python", "fastapi"],
            use_case="Building a virtual court app",
            confidence=0.85,
        )
        
        assert intent.primary_need == ResourceType.API
        assert len(intent.secondary_needs) == 1
        assert intent.confidence == 0.85
    
    def test_intent_confidence_bounds(self):
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            Intent(
                primary_need=ResourceType.API,
                use_case="Test",
                confidence=1.5,  # Invalid
            )
        
        with pytest.raises(ValidationError):
            Intent(
                primary_need=ResourceType.API,
                use_case="Test",
                confidence=-0.1,  # Invalid
            )


class TestRankingScore:
    """Tests for RankingScore model."""
    
    def test_ranking_score_creation(self):
        """Test creating a ranking score with valid values."""
        score = RankingScore(
            resource_id=uuid4(),
            semantic_relevance=0.8,
            popularity=0.6,
            optimization=0.7,
            freshness=0.9,
            final_score=0.75,
        )
        
        assert score.semantic_relevance == 0.8
        assert score.final_score == 0.75
    
    def test_ranking_score_bounds(self):
        """Test that all scores must be between 0 and 1."""
        with pytest.raises(ValidationError):
            RankingScore(
                resource_id=uuid4(),
                semantic_relevance=1.5,  # Invalid
                popularity=0.6,
                optimization=0.7,
                freshness=0.9,
                final_score=0.75,
            )


class TestUserContext:
    """Tests for UserContext model."""
    
    def test_user_context_creation(self):
        """Test creating user context."""
        context = UserContext(
            user_id=uuid4(),
            preferred_language="en",
            tech_stack=["python", "react"],
            search_history=["virtual court API", "ML models"],
        )
        
        assert context.preferred_language == "en"
        assert len(context.tech_stack) == 2
        assert len(context.search_history) == 2


class TestBoilerplatePackage:
    """Tests for BoilerplatePackage model."""
    
    def test_boilerplate_package_creation(self):
        """Test creating a boilerplate package."""
        resource = Resource(
            id=uuid4(),
            type=ResourceType.API,
            name="Test API",
            description="A test API",
            pricing_type=PricingType.FREE,
            source_url="https://example.com",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        code_file = CodeFile(
            path="src/main.py",
            content="print('Hello')",
            language="python",
        )
        
        package = BoilerplatePackage(
            package_id="test-123",
            resources=[resource],
            language="python",
            files=[code_file],
            readme="# Test Package",
            env_template="API_KEY=your_key",
            created_at=datetime.now(),
            download_url="https://example.com/download",
        )
        
        assert package.language == "python"
        assert len(package.resources) == 1
        assert len(package.files) == 1
    
    def test_boilerplate_package_invalid_language(self):
        """Test that invalid languages are rejected."""
        resource = Resource(
            id=uuid4(),
            type=ResourceType.API,
            name="Test API",
            description="A test API",
            pricing_type=PricingType.FREE,
            source_url="https://example.com",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        with pytest.raises(ValidationError) as exc_info:
            BoilerplatePackage(
                package_id="test-123",
                resources=[resource],
                language="ruby",  # Not supported
                files=[],
                readme="# Test",
                env_template="",
                created_at=datetime.now(),
                download_url="https://example.com/download",
            )
        
        assert "language" in str(exc_info.value)
    
    def test_boilerplate_package_empty_resources(self):
        """Test that packages must have at least one resource."""
        with pytest.raises(ValidationError) as exc_info:
            BoilerplatePackage(
                package_id="test-123",
                resources=[],  # Empty
                language="python",
                files=[],
                readme="# Test",
                env_template="",
                created_at=datetime.now(),
                download_url="https://example.com/download",
            )
        
        assert "At least one resource" in str(exc_info.value)


class TestSearchResults:
    """Tests for SearchResults model."""
    
    def test_search_results_creation(self):
        """Test creating search results."""
        resource = Resource(
            id=uuid4(),
            type=ResourceType.API,
            name="Test API",
            description="A test API",
            pricing_type=PricingType.FREE,
            source_url="https://example.com",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        score = RankingScore(
            resource_id=resource.id,
            semantic_relevance=0.8,
            popularity=0.6,
            optimization=0.7,
            freshness=0.9,
            final_score=0.75,
        )
        
        result = SearchResult(
            resource=resource,
            score=score,
            explanation="Highly relevant API",
            snippet="A test API for testing",
        )
        
        intent = Intent(
            primary_need=ResourceType.API,
            use_case="Testing",
            confidence=0.9,
        )
        
        results = SearchResults(
            query="test API",
            intent=intent,
            results=[result],
            total_count=1,
            execution_time_ms=150.5,
            filters_applied=SearchFilters(),
        )
        
        assert results.query == "test API"
        assert len(results.results) == 1
        assert results.total_count == 1
        assert results.execution_time_ms == 150.5
