import os
import boto3
import json
from dotenv import load_dotenv
from clients.opensearch import OpenSearchClient

load_dotenv()

# Config
APP_REGION = os.getenv("AWS_REGION", "ap-northeast-3") # Osaka
AI_REGION = os.getenv("BEDROCK_REGION", "ap-northeast-1") # Tokyo
MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
EMBED_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1")

# Clients - Pinning Bedrock to Tokyo
bedrock = boto3.client(service_name='bedrock-runtime', region_name=AI_REGION)
os_client = OpenSearchClient()

def test_rag_flow():
    print(f"\n--- 🕵️ App in {APP_REGION} | Bedrock in {AI_REGION} ---")
    
    try:
        # 1. Embeddings
        print(f"1. Generating Embedding via {EMBED_ID}...")
        body = json.dumps({"inputText": "Hindi NLP models"})
        res = bedrock.invoke_model(body=body, modelId=EMBED_ID)
        vector = json.loads(res.get('body').read())['embedding']
        print(f"   ✅ Vector success ({len(vector)} dims)")

        # 2. Search (Pinecone is region-agnostic)
        print("2. Querying Pinecone...")
        search_res = os_client.search(body={"query": {"knn": {"embedding": {"vector": vector, "k": 2}}}})
        print(f"   ✅ Found {len(search_res['hits']['hits'])} tools.")

        # 3. Chat
        print(f"3. Calling Claude (Tokyo Profile)...")
        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": "Recommend one Hindi model."}]}],
            system=[{"text": os.getenv("SYSTEM_PROMPT")}]
        )
        print(f"\n🤖 AI: {response['output']['message']['content'][0]['text']}\n")
        print("--- ✨ HYBRID TEST SUCCESSFUL ---")

    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_rag_flow()
