"""
Test script to verify pipeline imports work correctly.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing imports...")

try:
    print("1. Importing stages...")
    from ingestion.stages import (
        FetchStage,
        NormalizeStage,
        DedupeStage,
        UpsertStage,
        EmbeddingStage,
        IndexingStage,
        RankingStage,
        CacheStage,
    )
    print("   ✓ All stages imported successfully")

    print("\n2. Importing pipeline...")
    from ingestion.pipeline import IngestionPipeline, run_ingestion
    print("   ✓ Pipeline imported successfully")

    print("\n3. Importing services...")
    from ingestion.services.lock_service import LockService
    from ingestion.services.run_tracker import RunTracker
    from ingestion.services.snapshot_service import SnapshotService
    print("   ✓ All services imported successfully")

    print("\n4. Importing repository...")
    from ingestion.repository import IngestionRepository
    print("   ✓ Repository imported successfully")

    print("\n5. Importing normalization...")
    from ingestion.normalization import canonicalize_resource
    print("   ✓ Normalization imported successfully")

    print("\n6. Importing models...")
    from models import CanonicalResource, IngestionSource, IngestionStatus
    print("   ✓ Models imported successfully")

    print("\n" + "=" * 70)
    print("✅ ALL IMPORTS SUCCESSFUL!")
    print("=" * 70)
    print("\nThe pipeline is ready to use.")
    print("\nNext steps:")
    print("1. Set up infrastructure (PostgreSQL, Redis, OpenSearch, Bedrock)")
    print("2. Run: python scripts/run_full_backfill.py")

except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
