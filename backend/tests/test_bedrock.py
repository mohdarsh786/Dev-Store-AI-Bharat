"""Test Bedrock connection and embedding generation"""
import logging
from clients.bedrock import BedrockClient
from services.embeddings import get_embeddings_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_bedrock_client():
    """Test basic Bedrock client"""
    try:
        logger.info("Testing Bedrock client...")
        client = BedrockClient()
        
        # Test embedding generation
        logger.info("Generating test embedding...")
        embedding = client.generate_embedding("Hello, this is a test")
        logger.info(f"✅ Embedding generated successfully! Dimensions: {len(embedding)}")
        
        # Test health check
        health = client.health_check()
        logger.info(f"Health check: {health}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Bedrock client test failed: {e}", exc_info=True)
        return False

def test_embeddings_service():
    """Test embeddings service"""
    try:
        logger.info("\nTesting Embeddings Service...")
        service = get_embeddings_service(model_version="v2")
        
        # Test single embedding
        logger.info("Generating embedding with service...")
        embedding = service.get_embedding("Test query for semantic search")
        logger.info(f"✅ Service embedding generated! Dimensions: {len(embedding)}")
        
        # Test resource embedding
        logger.info("Generating resource embedding...")
        resource = {
            'name': 'Test API',
            'description': 'A test API for demonstration',
            'tags': ['test', 'api'],
            'resource_type': 'API'
        }
        resource_embedding = service.get_resource_embedding(resource)
        logger.info(f"✅ Resource embedding generated! Dimensions: {len(resource_embedding)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Embeddings service test failed: {e}", exc_info=True)
        return False

def main():
    logger.info("=" * 60)
    logger.info("Bedrock & Embeddings Service Test")
    logger.info("=" * 60)
    
    # Test Bedrock client
    bedrock_ok = test_bedrock_client()
    
    # Test embeddings service
    embeddings_ok = test_embeddings_service()
    
    logger.info("\n" + "=" * 60)
    logger.info("Test Results:")
    logger.info(f"  Bedrock Client: {'✅ PASS' if bedrock_ok else '❌ FAIL'}")
    logger.info(f"  Embeddings Service: {'✅ PASS' if embeddings_ok else '❌ FAIL'}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
