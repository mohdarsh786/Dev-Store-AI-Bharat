import os
import psycopg2
import boto3
import json
from pinecone import Pinecone
from dotenv import load_dotenv
import time

load_dotenv()

# Setup Clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1') # Update region if needed
conn = psycopg2.connect(os.getenv("DATABASE_URL")) # Neon Connection

def get_embedding(text):
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        body=body, 
        modelId="amazon.titan-embed-text-v1", 
        accept="application/json", 
        contentType="application/json"
    )
    return json.loads(response.get('body').read())['embedding']

def start_sync():
    cur = conn.cursor()
    
    # 1. Fetch only records that are NOT in the first 900
    # Hum 'id' se sort karke OFFSET use karenge
    print("📡 Fetching the remaining 886 records from Neon...")
    cur.execute("""
        SELECT id, name, description, category, source_url, stars, downloads 
        FROM resources 
        ORDER BY id ASC 
        OFFSET 900
    """)
    rows = cur.fetchall()
    
    print(f"🚀 Found {len(rows)} records. Starting Ingestion...")

    batch_size = 20 # Safety for API limits
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        upsert_data = []

        for row in batch:
            try:
                # Combine name + description for better search context
                content_to_embed = f"{row[1]}: {row[2]}"
                vector = get_embedding(content_to_embed)

                upsert_data.append({
                    "id": str(row[0]),
                    "values": vector,
                    "metadata": {
                        "name": str(row[1]),
                        "description": str(row[2]),
                        "category": str(row[3]),
                        "source_url": str(row[4]),
                        "stars": int(row[5] or 0),
                        "downloads": int(row[6] or 0)
                    }
                })
            except Exception as e:
                print(f"❌ Error embedding ID {row[0]}: {e}")

        if upsert_data:
            index.upsert(vectors=upsert_data)
            print(f"✅ Batched {i + len(batch)} / {len(rows)} records...")
        
        time.sleep(0.5) # Avoid hitting Bedrock/Pinecone rate limits

    print("\n🔥 MISSION COMPLETE! Pinecone is now fully synced with 1786 records.")

if __name__ == "__main__":
    start_sync()
