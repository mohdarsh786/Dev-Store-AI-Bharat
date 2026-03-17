"""
Partial backfill for top 10 resources by rank_score.
Ensures type, is_free, and insights are explicitly stored in Pinecone.
"""
import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from pinecone import Pinecone
from clients.database import DatabaseClient
from clients.bedrock import BedrockClient

load_dotenv()

CATEGORY_MAP = {'model': 'model', 'api': 'api', 'dataset': 'dataset'}

def partial_backfill_top10():
    db = DatabaseClient()
    bedrock = BedrockClient()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    print("Running partial backfill for top 10 resources by rank_score...")
    res_list = db.execute_query(
        "SELECT * FROM resources ORDER BY rank_score DESC NULLS LAST LIMIT 10"
    )
    print(f"Found {len(res_list)} resources.")

    vectors = []
    for res in res_list:
        try:
            raw_type = str(res.get('type') or 'api').lower()
            category = CATEGORY_MAP.get(raw_type, raw_type)
            pricing = res.get('pricing_type') or 'free'
            is_free = str(pricing).lower() == 'free'
            description = str(res.get('description') or '')
            insights = description[:200]
            
            metadata = {
                "resource_id": str(res.get('id')),
                "name": res.get('name', ''),
                "category": category,
                "type": category,
                "is_free": is_free,
                "pricing_type": pricing,
                "stars": int(res.get('github_stars') or res.get('stars') or 0),
                "downloads": int(res.get('download_count') or res.get('downloads') or 0),
                "insights": insights,
                "code_snippet": str(res.get('code_snippet', '')),
                "text_content": description
            }

            text_content = description.strip() or metadata['name']
            text_to_embed = f"{metadata['name']} {metadata['category']} {text_content}"

            vector = None
            for retry in range(5):
                try:
                    vector = bedrock.generate_embedding(text_to_embed)
                    break
                except Exception as e:
                    if "Throttling" in str(e) or "429" in str(e):
                        wait = 5 * (2 ** retry)
                        print(f"Throttled. Waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise e

            if not vector or not any(vector) or np.all(np.array(vector) == 0):
                print(f"[SKIP] Zero-vector for: {metadata['name']}")
                continue

            vectors.append({
                "id": f"{metadata['resource_id']}-full",
                "values": vector,
                "metadata": metadata
            })
            print(f"  Prepared: {metadata['name']} (type={category}, is_free={is_free})")
            time.sleep(1.0)

        except Exception as e:
            print(f"Warning: Error on {res.get('name')}: {e}")
            continue

    if vectors:
        index.upsert(vectors=vectors)
        print(f"SUCCESS: Upserted {len(vectors)} vectors for top 10 resources.")
    else:
        print("WARNING: No vectors to upsert.")

if __name__ == "__main__":
    partial_backfill_top10()
