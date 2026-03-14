import os
import psycopg2
import boto3
import json
from pinecone import Pinecone
from dotenv import load_dotenv
import time

load_dotenv()

# Setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1') 
conn = psycopg2.connect(os.getenv("DATABASE_URL"))

def get_embedding(text):
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        body=body, 
        modelId="amazon.titan-embed-text-v1", 
        accept="application/json", 
        contentType="application/json"
    )
    return json.loads(response.get('body').read())['embedding']

def start_full_sync():
    cur = conn.cursor()
    
    # KOI OFFSET NAHI - Saara data uthayenge
    print("📡 Fetching ALL 1786 records from Neon to ensure 100% sync...")
    query = "SELECT id, name, description, type, source_url, github_stars, download_count, tags FROM resources ORDER BY id ASC"
    cur.execute(query)
    rows = cur.fetchall()
    
    total = len(rows)
    print(f"🚀 Starting Full Ingestion. Total: {total} records.")

    batch_size = 15 
    for i in range(0, total, batch_size):
        batch = rows[i:i + batch_size]
        upsert_data = []

        for row in batch:
            try:
                tags_str = ", ".join(row[7]) if isinstance(row[7], list) else ""
                content_to_embed = f"Name: {row[1]}. Description: {row[2]}. Tags: {tags_str}"
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
                        "downloads": int(row[6] or 0),
                        "tags": tags_str
                    }
                })
            except Exception as e:
                print(f"❌ Error on ID {row[0]}: {e}")

        if upsert_data:
            index.upsert(vectors=upsert_data)
            print(f"✅ Progress: {i + len(batch)} / {total} processed...")
        
        time.sleep(1) # Cooldown for Bedrock

    print("\n🔥 FINAL MISSION COMPLETE! Count must be 1786 now.")

if __name__ == "__main__":
    start_full_sync()
