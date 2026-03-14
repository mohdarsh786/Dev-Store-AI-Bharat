import os
import sys
import boto3
import json
import logging
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Clients
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
db_url = os.getenv("DATABASE_URL")

pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

def generate_embedding(data_text):
    # This model outputs EXACTLY 1536 dimensions
    body = json.dumps({"inputText": data_text[:2000]}) 
    response = bedrock.invoke_model(
        body=body, 
        modelId='amazon.titan-embed-text-v1', 
        accept='application/json', 
        contentType='application/json'
    )
    return json.loads(response.get('body').read())['embedding']

def migrate():
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        logger.info("📡 Fetching records from Neon...")
        query = text("SELECT * FROM resources")
        results = [dict(row) for row in db.execute(query).mappings().all()]
        logger.info(f"✅ Found {len(results)} records.")
    finally:
        db.close()
        engine.dispose()

    logger.info("🧠 Processing high-quality 1536-dim embeddings...")
    vectors = []
    
    for i, row in enumerate(results):
        try:
            name = row.get('name') or 'Unknown'
            desc = row.get('description') or ''
            rid = str(row.get('id'))
            
            vector = generate_embedding(f"{name} {desc}")
            
            metadata = {
                "name": str(name),
                "description": str(desc),
                "category": str(row.get('category') or 'api'),
                "stars": int(row.get('stars') or 0),
                "downloads": int(row.get('downloads') or 0),
                "source_url": str(row.get('source_url') or '')
            }
            
            vectors.append((rid, vector, metadata))
            logger.info(f"[{i+1}/{len(results)}] Prepared: {name}")
            
            # Rate limit protection
            time.sleep(0.2) 

            if len(vectors) >= 50:
                index.upsert(vectors=vectors)
                logger.info(f"📤 Uploaded batch of {len(vectors)}")
                vectors = []

        except Exception as e:
            logger.error(f"⚠️ Error on {name}: {e}")
            time.sleep(5) # Cooldown for AWS

    if vectors:
        index.upsert(vectors=vectors)
        logger.info("✨ SUCCESS! Migration complete.")

if __name__ == "__main__":
    migrate()
