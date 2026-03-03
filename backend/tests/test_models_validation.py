"""
Tests for data model validation rules.
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
    Intent,
    RankingScore,
    BoilerplatePackage,
    CodeFile,
)


def test_resource_name_validation():
    """Test that Resource name must not be empty."""
    with pytest.raises(ValidationError) as exc_info:
        Resource(
            id=uuid4(),
            type=ResourceType.API,
            name="",  # Empty name should fail
            description="A test API",
            pricing_type=PricingType.FREE,
            source_url="https://example.com/api",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
    assert "name" in str(exc_info.value).lower()


def test_resource_url_validation():
    """Test that Resource URLs must be valid HTTP(S) URLs."""
    with pytest.raises(ValidationError) as exc_info:
        Resource(
            id=uuid4(),
            type=ResourceType.API,
            name="Test API",
            description="A test API",
            pricing_type=PricingType.FREE,
            source_url="not-a-valid-url",  # Invalid URL
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
    assert "source_url" in str(exc_info.value).lower()


def test_resource_github_stars_validation():
    """Test that github_stars must be non-negative."""
    with pytest.raises(ValidationError) as exc_info:
        Resource(
            id=uuid4(),
            type=ResourceType.API,
            name="Test API",
            description="A test API",
            pricing_type=PricingType.FREE,
            source_url="https://example.com/api",
            github_stars=-10,  # Negative stars should fail
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
    assert "github_stars" in str(exc_info.value).lower()


def test_resource_embedding_dimension_validation():
    """Test that embedding must have exactly 1536 dimensions."""
    with pytest.raises(ValidationError) as exc_info:
        Resource(
            id=uuid4(),
            type=ResourceType.API,
            name="Test API",
            description="A test API",
            pricing_type=PricingType.FREE,
            source_url="https://example.com/api",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            embedding=[0.1] * 100,  # Wrong dimension (should be 1536)
        )
    assert "1536" in str(exc_info.value)


def test_intent_confidence_validation():
    """Test that Intent confidence must be between 0 and 1."""
    with pytest.raises(ValidationError) as exc_info:
        Intent(
            primary_need=ResourceType.API,
            use_case="Testing",
            confidence=1.5,  # > 1.0 should fail
        )
    assert "confidence" in str(exc_info.value).lower()
    
    with pytest.raises(ValidationError) as exc_info:
        Intent(
            primary_need=ResourceType.API,
            use_case="Testing",
            confidence=-0.1,  # < 0.0 should fail
        )
    assert "confidence" in str(exc_info.value).lower()


def test_ranking_score_bounds_validation():
    """Test that all RankingScore components must be between 0 and 1."""
    # Test semantic_relevance > 1
    with pytest.raises(ValidationError) as exc_info:
        RankingScore(
            resource_id=uuid4(),
            semantic_relevance=1.5,  # > 1.0 should fail
            popularity=0.6,
            optimization=0.7,
            freshness=0.9,
            final_score=0.75,
        )
    assert "semantic_relevance" in str(exc_info.value).lower()
    
    # Test popularity < 0
    with pytest.raises(ValidationError) as exc_info:
        RankingScore(
            resource_id=uuid4(),
            semantic_relevance=0.8,
            popularity=-0.1,  # < 0.0 should fail
            optimization=0.7,
            freshness=0.9,
            final_score=0.75,
        )
    assert "popularity" in str(exc_info.value).lower()


def test_boilerplate_language_validation():
    """Test that BoilerplatePackage language must be python, javascript, or typescript."""
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
        content="# Code",
        language="python",
    )
    
    with pytest.raises(ValidationError) as exc_info:
        BoilerplatePackage(
            package_id="test-123",
            resources=[resource],
            language="java",  # Not in allowed list
            files=[code_file],
            readme="# README",
            env_template="KEY=value",
            created_at=datetime.now(),
            download_url="https://example.com/download.zip",
        )
    assert "language" in str(exc_info.value).lower()


def test_boilerplate_resources_not_empty():
    """Test that BoilerplatePackage must have at least one resource."""
    code_file = CodeFile(
        path="src/main.py",
        content="# Code",
        language="python",
    )
    
    with pytest.raises(ValidationError) as exc_info:
        BoilerplatePackage(
            package_id="test-123",
            resources=[],  # Empty resources should fail
            language="python",
            files=[code_file],
            readme="# README",
            env_template="KEY=value",
            created_at=datetime.now(),
            download_url="https://example.com/download.zip",
        )
    assert "at least one resource" in str(exc_info.value).lower()


def test_code_file_path_not_empty():
    """Test that CodeFile path must not be empty."""
    with pytest.raises(ValidationError) as exc_info:
        CodeFile(
            path="",  # Empty path should fail
            content="# Code",
            language="python",
        )
    assert "path" in str(exc_info.value).lower()


def test_valid_embedding_dimension():
    """Test that embedding with correct dimension (1536) is accepted."""
    resource = Resource(
        id=uuid4(),
        type=ResourceType.API,
        name="Test API",
        description="A test API",
        pricing_type=PricingType.FREE,
        source_url="https://example.com/api",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        embedding=[0.1] * 1536,  # Correct dimension
    )
    assert len(resource.embedding) == 1536


def test_valid_boilerplate_languages():
    """Test that all valid languages are accepted for BoilerplatePackage."""
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
        content="# Code",
        language="python",
    )
    
    for lang in ["python", "javascript", "typescript"]:
        package = BoilerplatePackage(
            package_id=f"test-{lang}",
            resources=[resource],
            language=lang,
            files=[code_file],
            readme="# README",
            env_template="KEY=value",
            created_at=datetime.now(),
            download_url="https://example.com/download.zip",
        )
        assert package.language == lang
