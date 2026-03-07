"""
Quick test script to validate AWS IAM authentication

This script tests:
1. AWS credential resolution
2. OpenSearch connection with SigV4 signing
3. Bedrock Titan v2 embedding generation
4. Claude 3.5 Sonnet invocation
"""
import sys
import logging
from aws_client import AWSClient, AWSClientError
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_credentials():
    """Test AWS credential resolution"""
    print("\n" + "=" * 60)
    print("TEST 1: AWS Credentials")
    print("=" * 60)
    
    try:
        import boto3
        session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        
        print(f"✅ Credentials Valid")
        print(f"   Account: {identity['Account']}")
        print(f"   User ARN: {identity['Arn']}")
        print(f"   User ID: {identity['UserId']}")
        return True
        
    except Exception as e:
        print(f"❌ Credential test failed: {e}")
        return False


def test_opensearch_connection(client: AWSClient):
    """Test OpenSearch connection"""
    print("\n" + "=" * 60)
    print("TEST 2: OpenSearch Connection")
    print("=" * 60)
    
    try:
        # Check if index exists
        index_exists = client.opensearch_client.indices.exists(
            index=settings.opensearch_index_name
        )
        
        print(f"✅ OpenSearch Connected")
        print(f"   Host: {settings.opensearch_host}")
        print(f"   Index: {settings.opensearch_index_name}")
        print(f"   Index Exists: {index_exists}")
        
        # Get document count if index exists
        if index_exists:
            count = client.opensearch_client.count(
                index=settings.opensearch_index_name
            )
            print(f"   Document Count: {count.get('count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenSearch connection failed: {e}")
        return False


def test_bedrock_embedding(client: AWSClient):
    """Test Bedrock Titan v2 embedding generation"""
    print("\n" + "=" * 60)
    print("TEST 3: Bedrock Titan v2 Embeddings")
    print("=" * 60)
    
    try:
        test_text = "machine learning dataset for sentiment analysis"
        embedding = client.generate_embedding(test_text)
        
        print(f"✅ Embedding Generated")
        print(f"   Model: amazon.titan-embed-text-v2:0")
        print(f"   Input: '{test_text}'")
        print(f"   Dimensions: {len(embedding)}")
        print(f"   Sample values: {embedding[:5]}")
        
        if len(embedding) != 1024:
            print(f"⚠️  Warning: Expected 1024 dimensions, got {len(embedding)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False


def test_claude_invocation(client: AWSClient):
    """Test Claude 3.5 Sonnet invocation"""
    print("\n" + "=" * 60)
    print("TEST 4: Claude 3.5 Sonnet")
    print("=" * 60)
    
    try:
        messages = [
            {
                "role": "user",
                "content": "Say 'Hello from Claude!' in exactly 5 words."
            }
        ]
        
        response = client.invoke_claude(
            messages=messages,
            max_tokens=50,
            temperature=0.7
        )
        
        print(f"✅ Claude Invoked Successfully")
        print(f"   Model: anthropic.claude-3-5-sonnet-20240620-v1:0")
        print(f"   Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Claude invocation failed: {e}")
        return False


def test_end_to_end(client: AWSClient):
    """Test end-to-end RAG flow"""
    print("\n" + "=" * 60)
    print("TEST 5: End-to-End RAG Flow")
    print("=" * 60)
    
    try:
        # 1. Generate embedding
        query = "python web framework"
        print(f"Query: '{query}'")
        
        embedding = client.generate_embedding(query)
        print(f"✅ Step 1: Generated {len(embedding)}D embedding")
        
        # 2. Search OpenSearch
        results = client.knn_search(
            query_vector=embedding,
            k=3
        )
        print(f"✅ Step 2: Found {len(results)} results")
        
        if results:
            for i, result in enumerate(results[:3], 1):
                doc = result['document']
                print(f"   {i}. {doc.get('name', 'Unknown')} (score: {result['score']:.2f})")
        
        # 3. Generate response with Claude
        if results:
            context = "\n".join([
                f"{r['document'].get('name')}: {r['document'].get('description', '')[:100]}"
                for r in results[:2]
            ])
            
            messages = [
                {
                    "role": "user",
                    "content": f"Based on these resources:\n{context}\n\nAnswer: {query}"
                }
            ]
            
            answer = client.invoke_claude(messages, max_tokens=200)
            print(f"✅ Step 3: Generated response")
            print(f"   Answer: {answer[:150]}...")
        else:
            print(f"⚠️  Step 3: Skipped (no results to process)")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("AWS IAM AUTHENTICATION TEST SUITE")
    print("=" * 60)
    print(f"Region: {settings.aws_region}")
    print(f"OpenSearch: {settings.opensearch_host}")
    print("=" * 60)
    
    results = {
        "credentials": False,
        "opensearch": False,
        "embeddings": False,
        "claude": False,
        "end_to_end": False
    }
    
    # Test 1: Credentials
    results["credentials"] = test_credentials()
    
    if not results["credentials"]:
        print("\n❌ CRITICAL: Credentials test failed. Cannot proceed.")
        return 1
    
    # Initialize AWS client
    try:
        client = AWSClient(
            region_name=settings.aws_region,
            opensearch_host=settings.opensearch_host,
            opensearch_index=settings.opensearch_index_name,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )
    except Exception as e:
        print(f"\n❌ CRITICAL: Failed to initialize AWSClient: {e}")
        return 1
    
    # Test 2: OpenSearch
    results["opensearch"] = test_opensearch_connection(client)
    
    # Test 3: Embeddings
    results["embeddings"] = test_bedrock_embedding(client)
    
    # Test 4: Claude
    results["claude"] = test_claude_invocation(client)
    
    # Test 5: End-to-end (only if index has data)
    if results["opensearch"]:
        try:
            count = client.opensearch_client.count(
                index=settings.opensearch_index_name
            )
            if count.get('count', 0) > 0:
                results["end_to_end"] = test_end_to_end(client)
            else:
                print("\n⚠️  Skipping end-to-end test (index is empty)")
        except:
            print("\n⚠️  Skipping end-to-end test (index check failed)")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    total_tests = len([r for r in results.values() if r is not False])
    passed_tests = sum(results.values())
    
    print(f"\nPassed: {passed_tests}/{total_tests}")
    print("=" * 60)
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Your AWS IAM setup is working correctly.")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED. Review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
