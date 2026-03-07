"""
Simple connection test for OpenSearch and Bedrock
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import boto3
from config import settings
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

print("=" * 60)
print("TESTING AWS CONNECTIONS")
print("=" * 60)

# Test 1: AWS Credentials
print("\n1. Testing AWS Credentials...")
try:
    session = boto3.Session(
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region
    )
    sts = session.client('sts')
    identity = sts.get_caller_identity()
    print(f"✅ Credentials Valid")
    print(f"   Account: {identity['Account']}")
    print(f"   User: {identity['Arn']}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 2: Bedrock Connection
print("\n2. Testing Bedrock (Titan v2 Embeddings)...")
try:
    bedrock = session.client('bedrock-runtime')
    
    import json
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({
            "inputText": "test connection",
            "dimensions": 1024,
            "normalize": True
        })
    )
    
    result = json.loads(response['body'].read())
    embedding = result.get('embedding', [])
    
    print(f"✅ Bedrock Connected")
    print(f"   Model: amazon.titan-embed-text-v2:0")
    print(f"   Embedding dimensions: {len(embedding)}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("\nManual fix:")
    print("1. Go to AWS Bedrock Console")
    print("2. Enable model access for Titan Embeddings v2")
    print("3. Wait 2-3 minutes for activation")

# Test 3: OpenSearch Connection
print("\n3. Testing OpenSearch Serverless...")
try:
    # Clean host
    host = settings.opensearch_host.replace("https://", "").replace("http://", "")
    
    # Create AWS4Auth
    credentials = session.get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        settings.aws_region,
        'aoss',  # OpenSearch Serverless
        session_token=credentials.token
    )
    
    # Create OpenSearch client
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30
    )
    
    # Test connection - check if index exists
    index_exists = client.indices.exists(index=settings.opensearch_index_name)
    
    print(f"✅ OpenSearch Connected")
    print(f"   Host: {host}")
    print(f"   Index: {settings.opensearch_index_name}")
    print(f"   Index exists: {index_exists}")
    
    if index_exists:
        count = client.count(index=settings.opensearch_index_name)
        print(f"   Documents: {count.get('count', 0)}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("\nManual fix:")
    print("1. Go to OpenSearch Serverless Console")
    print("2. Select your collection")
    print("3. Go to 'Data access' tab")
    print("4. Add this IAM user ARN to data access policy:")
    print(f"   {identity['Arn']}")
    print("5. Grant permissions: aoss:*")
    print("6. Wait 1-2 minutes for policy to apply")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
