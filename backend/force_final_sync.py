import os
import psycopg2
import boto3
import json
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1') 
conn = psycopg2.connect(os.getenv("DATABASE_URL"))

def get_embedding(text):
    # Truncate text strictly to 30,000 chars to pass Bedrock 50k limit easily
    safe_text = text[:30000] 
    body = json.dumps({"inputText": safe_text})
    response = bedrock.invoke_model(body=body, modelId="amazon.titan-embed-text-v1")
    return json.loads(response.get('body').read())['embedding']

def force_fix():
    cur = conn.cursor()
    print("🔍 Fetching missing records from Neon...")
    
    # IDs find karna jo Pinecone mein nahi hain
    cur.execute("SELECT id, name, description, type, source_url, github_stars, download_count, tags FROM resources")
    all_neon = {str(row[0]): row for row in cur.fetchall()}
    
    p_ids = set()
    for ids in index.list():
        p_ids.update(ids)
    
    missing = set(all_neon.keys()) - p_ids
    print(f"📍 Missing IDs found: {missing}")

    for mid in missing:
        row = all_neon[mid]
        try:
            print(f"🚀 Force syncing with truncation: {row[1]}")
            tags_str = ", ".join(row[7]) if isinstance(row[7], list) else ""
            content = f"Name: {row[1]}. Description: {row[2]}. Tags: {tags_str}"
            
            vector = get_embedding(content)
            index.upsert(vectors=[{
                "id": mid,
                "values": vector,
                "metadata": {
                    "name": str(row[1]),
                    "description": str(row[2])[:500], # Metadata safety
                    "category": str(row[3]),
                    "source_url": str(row[4]),
                    "stars": int(row[5] or 0),
                    "downloads": int(row[6] or 0)
                }
            }])
            print(f"✅ Fixed: {row[1]}")
        except Exception as e:
            print(f"❌ Still failing for {row[1]}: {e}")

if __name__ == "__main__":
    force_fix()
