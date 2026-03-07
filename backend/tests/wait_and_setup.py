"""
Wait for OpenSearch permissions and auto-setup index

This script will:
1. Test connections every 30 seconds
2. Once OpenSearch is accessible, create the index
3. Show you the status
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

import boto3
from config import settings
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

def test_opensearch():
    """Test if OpenSearch is accessible"""
    try:
        session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        
        host = settings.opensearch_host.replace("https://", "").replace("http://", "")
        credentials = session.get_credentials()
        
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            settings.aws_region,
            'aoss',
            session_token=credentials.token
        )
        
        client = OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=10
        )
        
        # Try to check if index exists
        client.indices.exists(index=settings.opensearch_index_name)
        return True, client
        
    except Exception as e:
        return False, str(e)

def create_index(client):
    """Create k-NN index"""
    try:
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    "name": {"type": "text"},
                    "description": {"type": "text"},
                    "category": {"type": "keyword"},
                    "source": {"type": "keyword"}
                }
            }
        }
        
        # Check if exists
        if client.indices.exists(index=settings.opensearch_index_name):
            print(f"✅ Index '{settings.opensearch_index_name}' already exists")
            return True
        
        # Create index
        response = client.indices.create(
            index=settings.opensearch_index_name,
            body=index_body
        )
        
        print(f"✅ Index '{settings.opensearch_index_name}' created successfully!")
        print("   - Vector field: 'embedding'")
        print("   - Dimensions: 1024 (Titan v2)")
        print("   - Algorithm: HNSW")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create index: {e}")
        return False

def main():
    print("=" * 60)
    print("WAITING FOR OPENSEARCH PERMISSIONS")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Open AWS Console: https://{}.console.aws.amazon.com/aos/".format(settings.aws_region))
    print("2. Go to your OpenSearch collection")
    print("3. Click 'Data access' tab")
    print("4. Add IAM user: arn:aws:iam::{}:user/devstore-service-user".format(os.getenv('AWS_ACCOUNT_ID', 'YOUR_ACCOUNT_ID')))
    print("5. Save and wait...")
    print("\nThis script will automatically detect when permissions are applied.")
    print("=" * 60)
    
    attempt = 1
    max_attempts = 20  # 10 minutes total
    
    while attempt <= max_attempts:
        print(f"\n[Attempt {attempt}/{max_attempts}] Testing OpenSearch connection...")
        
        success, result = test_opensearch()
        
        if success:
            print("✅ OpenSearch is now accessible!")
            print("\nCreating k-NN index...")
            
            if create_index(result):
                print("\n" + "=" * 60)
                print("🎉 SUCCESS! Everything is ready.")
                print("=" * 60)
                print("\nNext steps:")
                print("1. Run: python test_connections_simple.py")
                print("2. Prepare your JSON data files")
                print("3. Start ingesting data")
                return 0
            else:
                print("\n⚠️ Index creation failed. Try manually:")
                print("   python setup_opensearch_index.py")
                return 1
        else:
            print(f"❌ Still not accessible: {result}")
            
            if attempt < max_attempts:
                print(f"⏰ Waiting 30 seconds before retry...")
                time.sleep(30)
            
        attempt += 1
    
    print("\n" + "=" * 60)
    print("⏰ TIMEOUT: Permissions not applied after 10 minutes")
    print("=" * 60)
    print("\nPlease verify:")
    print("1. You added the correct IAM user ARN")
    print("2. You saved the data access policy")
    print("3. The collection is in 'Active' state")
    print("\nThen run this script again.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
