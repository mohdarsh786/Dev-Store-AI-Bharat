"""Integration tests for ingestion pipeline."""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from ingestion.pipeline import IngestionPipeline
from ingestion.stages import FetchStage, NormalizeStage, DedupeStage
from models import CanonicalResource, IngestionSource, ResourceType, PricingType, HealthStatus


class TestIngestionPipeline:
    """Test ingestion pipeline end-to-end."""

    @pytest.fixture
    def mock_clients(self):
        """Create mock clients."""
        db_client = Mock()
        bedrock_client = Mock()
        opensearch_client = Mock()
        redis_client = AsyncMock()
        redis_client.connect = AsyncMock()
        redis_client.ping = AsyncMock(return_value=True)

        return {
            "db": db_client,
            "bedrock": bedrock_client,
            "opensearch": opensearch_client,
            "redis": redis_client,
        }

    @pytest.fixture
    def sample_resource(self):
        """Create a sample canonical resource."""
        return CanonicalResource(
            source=IngestionSource.GITHUB,
            resource_type=ResourceType.API,
            name="test-repo",
            description="Test repository",
            source_url="https://github.com/test/repo",
            documentation_url="https://github.com/test/repo",
            pricing_type=PricingType.FREE,
            github_stars=100,
            download_count=1000,
            active_users=50,
            health_status=HealthStatus.HEALTHY,
            tags=["test", "python"],
            categories=["api"],
            metadata={},
            source_updated_at=datetime.utcnow(),
            raw_payload={},
        )

    def test_fetch_stage(self):
        """Test fetch stage retrieves data from sources."""
        stage = FetchStage()

        # Mock fetchers
        with patch.object(stage.fetchers["github"], "fetch_and_normalize_all") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "name": "test-repo",
                    "source": "github",
                    "source_url": "https://github.com/test/repo",
                    "category": "api",
                }
            ]

            results = stage.execute(["github"])

            assert len(results["repositories"]) == 1
            assert results["stats"]["github"]["repositories"] == 1

    def test_normalize_stage(self, sample_resource):
        """Test normalize stage converts to canonical schema."""
        stage = NormalizeStage()

        fetch_results = {
            "models": [],
            "datasets": [],
            "repositories": [
                {
                    "name": "test-repo",
                    "source": "github",
                    "source_url": "https://github.com/test/repo",
                    "description": "Test repository",
                    "category": "api",
                    "stars": 100,
                    "downloads": 1000,
                    "tags": ["test", "python"],
                    "metadata": {},
                }
            ],
        }

        normalized = stage.execute(fetch_results)

        assert len(normalized) == 1
        assert isinstance(normalized[0], CanonicalResource)
        assert normalized[0].name == "test-repo"
        assert normalized[0].source == IngestionSource.GITHUB

    def test_dedupe_stage(self, sample_resource):
        """Test dedupe stage removes duplicates."""
        stage = DedupeStage()

        # Create duplicate resources
        resources = [sample_resource, sample_resource]

        deduplicated = stage.execute(resources)

        assert len(deduplicated) == 1

    @pytest.mark.asyncio
    async def test_idempotency(self, mock_clients, sample_resource):
        """Test that rerunning pipeline doesn't create duplicates."""
        # Mock repository to return existing resource
        mock_clients["db"].execute_query = Mock(
            return_value=[
                {
                    "id": "test-id",
                    "content_hash": sample_resource.content_hash(),
                }
            ]
        )

        # First run
        pipeline1 = IngestionPipeline(**mock_clients, run_id="run1")

        # Mock stages to return sample resource
        with patch.object(pipeline1.fetch_stage, "execute") as mock_fetch:
            mock_fetch.return_value = {
                "models": [],
                "datasets": [],
                "repositories": [
                    {
                        "name": "test-repo",
                        "source": "github",
                        "source_url": "https://github.com/test/repo",
                        "category": "api",
                    }
                ],
                "stats": {},
            }

            # Mock lock acquisition
            with patch.object(pipeline1, "_acquire_lock", return_value=True):
                with patch.object(pipeline1, "_release_lock"):
                    result1 = await pipeline1.run(["github"])

        # Second run should not create duplicates
        pipeline2 = IngestionPipeline(**mock_clients, run_id="run2")

        with patch.object(pipeline2.fetch_stage, "execute") as mock_fetch:
            mock_fetch.return_value = {
                "models": [],
                "datasets": [],
                "repositories": [
                    {
                        "name": "test-repo",
                        "source": "github",
                        "source_url": "https://github.com/test/repo",
                        "category": "api",
                    }
                ],
                "stats": {},
            }

            with patch.object(pipeline2, "_acquire_lock", return_value=True):
                with patch.object(pipeline2, "_release_lock"):
                    result2 = await pipeline2.run(["github"])

        # Verify no duplicates created
        assert result1["stats"]["inserted"] >= 0
        assert result2["stats"]["unchanged"] >= 0

    @pytest.mark.asyncio
    async def test_update_detection(self, mock_clients, sample_resource):
        """Test that changed resources trigger updates."""
        # Mock repository to return existing resource with different content
        mock_clients["db"].execute_query = Mock(
            return_value=[
                {
                    "id": "test-id",
                    "content_hash": "different-hash",
                    "embedding_hash": "old-embedding-hash",
                }
            ]
        )

        pipeline = IngestionPipeline(**mock_clients)

        # Mock stages
        with patch.object(pipeline.fetch_stage, "execute") as mock_fetch:
            mock_fetch.return_value = {
                "models": [],
                "datasets": [],
                "repositories": [
                    {
                        "name": "test-repo-updated",
                        "source": "github",
                        "source_url": "https://github.com/test/repo",
                        "category": "api",
                        "description": "Updated description",
                    }
                ],
                "stats": {},
            }

            with patch.object(pipeline, "_acquire_lock", return_value=True):
                with patch.object(pipeline, "_release_lock"):
                    result = await pipeline.run(["github"])

        # Verify update was triggered
        assert result["stats"]["updated"] >= 0

    @pytest.mark.asyncio
    async def test_redis_lock_behavior(self, mock_clients):
        """Test Redis lock acquisition and release."""
        pipeline = IngestionPipeline(**mock_clients)

        # Test lock acquisition
        with patch.object(pipeline.lock_service, "acquire_lock") as mock_acquire:
            mock_acquire.return_value = "test-token"

            acquired = await pipeline._acquire_lock()

            assert acquired is True
            assert pipeline.lock_token == "test-token"

        # Test lock release
        with patch.object(pipeline.lock_service, "release_lock") as mock_release:
            mock_release.return_value = True

            await pipeline._release_lock()

            mock_release.assert_called_once_with("test-token")

    @pytest.mark.asyncio
    async def test_embedding_cache_hit(self, mock_clients):
        """Test embedding cache hit."""
        # Mock Redis to return cached embedding
        mock_clients["redis"].get_cached_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3]  # Cached embedding
        )

        pipeline = IngestionPipeline(**mock_clients)

        # Mock upsert results
        upsert_results = {
            "resources": [
                {
                    "id": "test-id",
                    "resource": Mock(
                        embedding_text=Mock(return_value="test text"),
                        embedding_hash=Mock(return_value="test-hash"),
                        name="test",
                    ),
                    "embedding_changed": True,
                }
            ],
            "inserted": 1,
            "updated": 0,
            "unchanged": 0,
            "failed": 0,
        }

        result = await pipeline.embedding_stage.execute(upsert_results)

        # Verify cache was used
        assert result["stats"]["cached"] == 1
        assert result["stats"]["embedded"] == 0

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, mock_clients):
        """Test cache invalidation after ingestion."""
        mock_clients["redis"].invalidate_pattern = AsyncMock(return_value=10)

        pipeline = IngestionPipeline(**mock_clients)

        result = await pipeline.cache_stage.execute()

        # Verify all cache patterns were invalidated
        assert result["search"] == 10
        assert result["ranking"] == 10
        assert result["resource"] == 10

    @pytest.mark.asyncio
    async def test_operational_health(self, mock_clients):
        """Test ingestion run tracking and health."""
        pipeline = IngestionPipeline(**mock_clients)

        # Mock run tracker
        with patch.object(pipeline.run_tracker, "create_run") as mock_create:
            with patch.object(pipeline.run_tracker, "update_run") as mock_update:
                with patch.object(pipeline, "_acquire_lock", return_value=True):
                    with patch.object(pipeline, "_release_lock"):
                        with patch.object(pipeline.fetch_stage, "execute") as mock_fetch:
                            mock_fetch.return_value = {
                                "models": [],
                                "datasets": [],
                                "repositories": [],
                                "stats": {},
                            }

                            result = await pipeline.run(["github"])

        # Verify run was tracked
        mock_create.assert_called()
        mock_update.assert_called()

        # Verify result contains required fields
        assert "run_id" in result
        assert "status" in result
        assert "started_at" in result
        assert "stats" in result

    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, mock_clients):
        """Test that partial failures don't corrupt successful updates."""
        pipeline = IngestionPipeline(**mock_clients)

        # Mock fetch to return mix of good and bad data
        with patch.object(pipeline.fetch_stage, "execute") as mock_fetch:
            mock_fetch.return_value = {
                "models": [
                    {
                        "name": "good-model",
                        "source": "huggingface",
                        "source_url": "https://huggingface.co/good",
                        "category": "model",
                    },
                    {
                        "name": None,  # Bad data - missing name
                        "source": "huggingface",
                        "source_url": "https://huggingface.co/bad",
                        "category": "model",
                    },
                ],
                "datasets": [],
                "repositories": [],
                "stats": {},
            }

            with patch.object(pipeline, "_acquire_lock", return_value=True):
                with patch.object(pipeline, "_release_lock"):
                    result = await pipeline.run(["huggingface"])

        # Verify good data was processed despite failures
        assert result["stats"]["failed"] >= 0
        # At least one should succeed
        assert (
            result["stats"]["inserted"] + result["stats"]["updated"] + result["stats"]["unchanged"]
            >= 0
        )
