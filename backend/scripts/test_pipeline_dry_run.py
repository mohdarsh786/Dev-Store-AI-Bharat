"""
Dry run test of the pipeline without infrastructure.

Tests that all components can be instantiated and basic flow works.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("PIPELINE DRY RUN TEST")
print("=" * 70)

try:
    print("\n1. Testing stage imports...")
    from ingestion.stages import (
        FetchStage,
        NormalizeStage,
        DedupeStage,
    )
    print("   ✓ Stages imported")

    print("\n2. Testing stage instantiation...")
    fetch_stage = FetchStage()
    normalize_stage = NormalizeStage()
    dedupe_stage = DedupeStage()
    print("   ✓ Stages instantiated")

    print("\n3. Testing fetch stage...")
    # Mock the fetchers to avoid actual API calls
    fetch_stage.fetchers = {
        "github": Mock(fetch_and_normalize_all=Mock(return_value=[])),
        "huggingface": Mock(fetch_and_normalize_all=Mock(return_value={"models": [], "datasets": []})),
        "kaggle": Mock(fetch_and_normalize_all=Mock(return_value=[])),
        "openrouter": Mock(fetch_and_normalize_all=Mock(return_value=[])),
    }
    
    result = fetch_stage.execute(["github"])
    print(f"   ✓ Fetch stage executed (fetched {len(result['repositories'])} repos)")

    print("\n4. Testing normalize stage...")
    normalized = normalize_stage.execute(result)
    print(f"   ✓ Normalize stage executed ({len(normalized)} resources)")

    print("\n5. Testing dedupe stage...")
    deduplicated = dedupe_stage.execute(normalized)
    print(f"   ✓ Dedupe stage executed ({len(deduplicated)} resources)")

    print("\n6. Testing models...")
    from models import CanonicalResource, IngestionSource, ResourceType, PricingType, HealthStatus
    from datetime import datetime
    
    test_resource = CanonicalResource(
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
        tags=["test"],
        categories=["api"],
        metadata={},
        source_updated_at=datetime.utcnow(),
        raw_payload={},
    )
    print(f"   ✓ CanonicalResource created: {test_resource.name}")
    print(f"   ✓ Unique key: {test_resource.unique_key}")
    print(f"   ✓ Content hash: {test_resource.content_hash()[:16]}...")

    print("\n7. Testing services...")
    from ingestion.services.lock_service import LockService
    from ingestion.services.snapshot_service import SnapshotService
    
    # Mock clients
    mock_redis = Mock()
    mock_s3 = Mock()
    
    lock_service = LockService(mock_redis)
    snapshot_service = SnapshotService(mock_s3, "test-bucket")
    print("   ✓ Services instantiated")

    print("\n" + "=" * 70)
    print("✅ DRY RUN SUCCESSFUL!")
    print("=" * 70)
    print("\nAll components are working correctly.")
    print("\nTo run with real infrastructure:")
    print("1. Set up PostgreSQL, Redis, OpenSearch, Bedrock")
    print("2. Configure environment variables")
    print("3. Run: python scripts/run_full_backfill.py")

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
