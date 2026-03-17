import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# Path setup MUST be before imports
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from pinecone import Pinecone
from clients.database import DatabaseClient
from clients.bedrock import BedrockClient

load_dotenv()

# --- CONFIGURATION ---
START_FROM = 0  # Change this to resume from a specific index (0-based)
EMBEDDING_PAUSE = 2.0  # Increased pause to avoid ThrottlingException
BATCH_SIZE = 50

def run_full_backfill():
    db = DatabaseClient()
    bedrock = BedrockClient()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    try:
        # Process ALL 1786 resources ordered by ID for consistent resume
        res_list = db.execute_query("SELECT * FROM resources ORDER BY id ASC")
        total_count = len(res_list)
        print(f"Starting full-database backfill for {total_count} resources.")
        if START_FROM > 0:
            print(f"Resuming from index: {START_FROM}")
        
        all_vectors = []
        for i, res in enumerate(res_list):
            if i < START_FROM:
                continue
                
            current_idx = i + 1
            if current_idx % 25 == 0:
                print(f"Progress: [{current_idx}/{total_count}]")
            
            try:
                # Multi-layer metadata mapping
                category = res.get('type') or 'api'
                if str(category).lower() == 'unknown':
                    name_lower = str(res.get('name', '')).lower()
                    if 'dataset' in name_lower: category = 'dataset'
                    elif 'api' in name_lower: category = 'api'
                    else: category = 'model'
                
                pricing = res.get('pricing_type') or 'free'
                is_free = (str(pricing).lower() == 'free')
                
                # Metadata construction
                metadata = {
                    "resource_id": str(res.get('id')),
                    "name": res.get('name', ''),
                    "category": category,
                    "type": category,
                    "is_free": is_free,
                    "pricing_type": pricing,
                    "stars": int(res.get('github_stars') or res.get('stars') or 0),
                    "downloads": int(res.get('download_count') or res.get('downloads') or 0),
                    "insights": str(res.get('description', ''))[:200],
                    "code_snippet": str(res.get('code_snippet', '')),
                    "text_content": str(res.get('description', ''))
                }

                # Empty String Check & Fallback
                text_content = metadata['text_content'].strip()
                if not text_content:
                    text_content = metadata['name']
                
                text_to_embed = f"{metadata['name']} {metadata['category']} {text_content}"
                
                # Generate embedding with retry for throttling
                vector = None
                max_retries = 8
                for retry in range(max_retries):
                    try:
                        vector = bedrock.generate_embedding(text_to_embed)
                        break
                    except Exception as e:
                        if "Throttling" in str(e) or "LimitExceeded" in str(e) or "429" in str(e):
                            wait_time = 5 * (2 ** retry)  # 5s, 10s, 20s, 40s...
                            print(f"Throttled at index {i}. Waiting {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"Bedrock Error at index {i}: {e}")
                            break
                
                if not vector:
                    print(f"Warning: [SKIP] Could not get embedding for Resource: {metadata['name']}")
                    continue
                
                # Zero-Vector Validation (Pinecone Error 400 Prevention)
                if not any(vector) or np.all(np.array(vector) == 0):
                    print(f"Warning: [SKIP] Zero-vector detected for Resource: {metadata['name']} (ID: {metadata['resource_id']})")
                    continue
                
                all_vectors.append({
                    "id": f"{metadata['resource_id']}-full",
                    "values": vector,
                    "metadata": metadata
                })

                # Small delay to keep constant pace
                time.sleep(EMBEDDING_PAUSE)

                # Batch Upsert
                if len(all_vectors) >= BATCH_SIZE:
                    index.upsert(vectors=all_vectors)
                    print(f"📤 Upserted batch ending at index {current_idx}. Cooling down for 10s...")
                    all_vectors = []
                    time.sleep(10) # 10-second cool-down after batch upsert

            except Exception as e:
                print(f"Warning: General error at index {i} ({res.get('name')}): {e}")
                continue

        # Final upsert for remaining items
        if all_vectors:
            index.upsert(vectors=all_vectors)
            print("✨ SUCCESS: Remaining items upserted.")

        print("🎉 SUCCESS: Full database backfill complete.")

    except Exception as e:
        print(f"❌ Critical Failure: {str(e)}")

if __name__ == "__main__":
    run_full_backfill()