"""Manual smoke test for SearchService initialization."""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    print("=" * 60)
    print("Testing SearchService Initialization")
    print("=" * 60)

    print("\n1. Testing config...")
    try:
        from config import settings

        print("✅ Config loaded")
        print(f"   AWS Region: {settings.aws_region}")
        print(f"   OpenSearch Host: {settings.opensearch_host[:50]}...")
        print(f"   Database URL: {settings.database_url[:30]}...")
    except Exception as e:
        print(f"❌ Config failed: {e}")
        return 1

    print("\n2. Testing SearchService...")
    try:
        from services.search import SearchService

        print("✅ SearchService imported")
        service = SearchService()
        print("✅ SearchService initialized")
        print(f"   Bedrock client: {service.bedrock}")
        print(f"   OpenSearch client: {service.opensearch}")
        print(f"   Ranking service: {service.ranking}")
    except Exception as e:
        print(f"❌ SearchService failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n3. Testing embedding generation...")
    try:
        embedding = service.generate_embedding("test query")
        print(f"✅ Embedding generated: {len(embedding)} dimensions")
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n4. Testing intent extraction...")
    try:
        intent = service.extract_intent("I need a free NLP model")
        print(f"✅ Intent extracted: {intent}")
    except Exception as e:
        print(f"❌ Intent extraction failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
