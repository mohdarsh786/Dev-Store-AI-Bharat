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
    # Aggrasive Truncation: 15,000 characters (approx 4,000 tokens)
    # Taaki 8,192 ki token limit se hum kaafi peeche rahein
    safe_text = text[:15000] 
    body = json.dumps({"inputText": safe_text})
    response = bedrock.invoke_model(body=body, modelId="amazon.titan-embed-text-v1")
    return json.loads(response.get('body').read())['embedding']

def final_push():
    cur = conn.cursor()
    ids_to_fix = ['0c8c3bcb-0afd-4605-b93e-a7468e7c9c3a', '32004515-07ef-4973-b9c5-00bd4f71c24a']
    
    for mid in ids_to_fix:
        cur.execute("SELECT id, name, description, type, source_url, github_stars, download_count, tags FROM resources WHERE id = %s", (mid,))
        row = cur.fetchone()
        if not row: continue

        try:
            print(f"🚀 Final Force Sync (Aggressive Truncation): {row[1]}")
            tags_str = ", ".join(row[7]) if isinstance(row[7], list) else ""
            # Only use Name + First part of description for these problematic records
            content = f"Name: {row[1]}. Description: {row[2]}"
            
            vector = get_embedding(content)
            index.upsert(vectors=[{
                "id": mid,
                "values": vector,
                "metadata": {
                    "name": str(row[1]),
                    "description": str(row[2])[:500],
                    "category": str(row[3]),
                    "source_url": str(row[4]),
                    "stars": int(row[5] or 0),
                    "downloads": int(row[6] or 0)
                }
            }])
            print(f"✅ SUCCESSFULLY FIXED: {row[1]}")
        except Exception as e:
            print(f"❌ Even extreme truncation failed for {row[1]}: {e}")

if __name__ == "__main__":
    final_push()
