"""
Basic tests for data models to verify they can be instantiated correctly.
"""

import pytest
from datetime import datetime
from uuid import uuid4
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


def test_resource_model_creation():
    """Test that Resource model can be created with valid data."""
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
    assert resource.name == "Test API"
    assert resource.type == ResourceType.API
    assert resource.pricing_type == PricingType.FREE


def test_search_filters_model():
    """Test that SearchFilters model can be created."""
    filters = SearchFilters(
        resource_types=[ResourceType.API, ResourceType.MODEL],
        pricing_types=[PricingType.FREE],
        min_stars=100,
    )
    assert len(filters.resource_types) == 2
    assert filters.pricing_types[0] == PricingType.FREE
    assert filters.min_stars == 100


def test_intent_model():
    """Test that Intent model can be created."""
    intent = Intent(
        primary_need=ResourceType.API,
        secondary_needs=[ResourceType.MODEL],
        tech_stack=["python", "fastapi"],
        use_case="Building a virtual court app",
        confidence=0.85,
    )
    assert intent.primary_need == ResourceType.API
    assert intent.confidence == 0.85
    assert "python" in intent.tech_stack


def test_ranking_score_model():
    """Test that RankingScore model can be created."""
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


def test_user_context_model():
    """Test that UserContext model can be created."""
    context = UserContext(
        user_id=uuid4(),
        preferred_language="en",
        tech_stack=["python", "react"],
        search_history=["machine learning API"],
    )
    assert context.preferred_language == "en"
    assert len(context.tech_stack) == 2


def test_code_file_model():
    """Test that CodeFile model can be created."""
    code_file = CodeFile(
        path="src/main.py",
        content="print('Hello, World!')",
        language="python",
    )
    assert code_file.path == "src/main.py"
    assert code_file.language == "python"


def test_boilerplate_package_model():
    """Test that BoilerplatePackage model can be created."""
    resource = Resource(
        id=uuid4(),
        type=ResourceType.API,
        name="Test API",
        description="A test API",
        pricing_type=PricingType.FREE,
        source_url="https://example.com/api",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    
    code_file = CodeFile(
        path="src/main.py",
        content="# API integration code",
        language="python",
    )
    
    package = BoilerplatePackage(
        package_id="test-package-123",
        resources=[resource],
        language="python",
        files=[code_file],
        readme="# Setup Instructions",
        env_template="API_KEY=your_key_here",
        created_at=datetime.now(),
        download_url="https://example.com/download/package.zip",
    )
    
    assert package.package_id == "test-package-123"
    assert package.language == "python"
    assert len(package.resources) == 1
    assert len(package.files) == 1


def test_search_results_model():
    """Test that SearchResults model can be created."""
    resource = Resource(
        id=uuid4(),
        type=ResourceType.API,
        name="Test API",
        description="A test API",
        pricing_type=PricingType.FREE,
        source_url="https://example.com/api",
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
        explanation="Highly relevant to your query",
        snippet="A test API for testing...",
    )
    
    intent = Intent(
        primary_need=ResourceType.API,
        use_case="Testing",
        confidence=0.9,
    )
    
    filters = SearchFilters()
    
    results = SearchResults(
        query="test API",
        intent=intent,
        results=[result],
        total_count=1,
        execution_time_ms=150.5,
        filters_applied=filters,
    )
    
    assert results.query == "test API"
    assert len(results.results) == 1
    assert results.total_count == 1
    assert results.execution_time_ms == 150.5
