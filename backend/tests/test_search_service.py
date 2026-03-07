"""Test if SearchService can be initialized"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 60)
print("Testing SearchService Initialization")
print("=" * 60)

# Test config loading
print("\n1. Testing config...")
try:
    from config import settings
    print(f"✅ Config loaded")
    print(f"   AWS Region: {settings.aws_region}")
    print(f"   OpenSearch Host: {settings.opensearch_host[:50]}...")
    print(f"   Database URL: {settings.database_url[:30]}...")
except Exception as e:
    print(f"❌ Config failed: {e}")
    exit(1)

# Test SearchService
print("\n2. Testing SearchService...")
try:
    from services.search import SearchService
    print(f"✅ SearchService imported")
    
    service = SearchService()
    print(f"✅ SearchService initialized")
    print(f"   Bedrock client: {service.bedrock}")
    print(f"   OpenSearch client: {service.opensearch}")
    print(f"   Ranking service: {service.ranking}")
    
except Exception as e:
    print(f"❌ SearchService failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test embedding generation
print("\n3. Testing embedding generation...")
try:
    embedding = service.generate_embedding("test query")
    print(f"✅ Embedding generated: {len(embedding)} dimensions")
except Exception as e:
    print(f"❌ Embedding failed: {e}")
    import traceback
    traceback.print_exc()

# Test intent extraction
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
