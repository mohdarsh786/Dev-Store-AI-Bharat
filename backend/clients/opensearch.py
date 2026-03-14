import os
import logging
import json
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class OpenSearchClient:
    def __init__(self, **kwargs):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        self._client = self 
        logger.info("🚀 Hijack: Universal Data Mapper Active")

    def search(self, *args, **kwargs):
        body = kwargs.get('body') or (args[0] if args else None)
        if not body: return {"hits": {"hits": [], "total": {"value": 0}}}
        
        try:
            query_obj = body.get('query', {})
            bool_obj = query_obj.get('bool', {})
            should_list = bool_obj.get('should', [])
            
            vector = None
            if should_list and isinstance(should_list, list):
                vector = should_list[0].get('knn', {}).get('embedding', {}).get('vector')
            
            if not vector:
                vector = query_obj.get('knn', {}).get('embedding', {}).get('vector')

            if not vector:
                return {"hits": {"hits": [], "total": {"value": 0}}}

            res = self.index.query(vector=vector, top_k=5, include_metadata=True)
            
            hits = []
            for m in res['matches']:
                meta = m['metadata']
                # MAPPING: Pinecone keys -> Backend expected keys
                source_data = {
                    "name": meta.get('name', 'Unknown'),
                    "description": meta.get('description', ''),
                    "resource_type": meta.get('category', 'tool'),
                    "category": meta.get('category', 'tool'),
                    "source": meta.get('source_url', ''), # Fixes KeyError: 'source'
                    "source_url": meta.get('source_url', ''),
                    "github_stars": meta.get('stars', 0), # Added for safety
                    "stars": meta.get('stars', 0),
                    "downloads": meta.get('downloads', 0)
                }
                
                hits.append({
                    "_id": m['id'],
                    "_score": 1.0,
                    "_source": source_data
                })
                
            return {"hits": {"hits": hits, "total": {"value": len(hits)}}}
            
        except Exception as e:
            logger.error(f"❌ Pinecone Error: {e}")
            return {"hits": {"hits": [], "total": {"value": 0}}}

    def knn_search(self, *args, **kwargs): return self.search(*args, **kwargs)
    def index_exists(self, *args, **kwargs): return True
    def count(self, *args, **kwargs): return {"count": 900}
