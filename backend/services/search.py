import os
import logging
from typing import List, Dict, Any
from pinecone import Pinecone
from clients.bedrock import BedrockClient
from services.ranking import RankingService

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self):
        self.bedrock = BedrockClient()
        self.ranking = RankingService()
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    def search(self, query: str, limit: int = 20, **kwargs) -> Dict[str, Any]:
        try:
            # 1. Vectorize query
            query_vector = self.bedrock.generate_embedding(query)
            
            # 2. Query Pinecone
            # We fetch top_k = 40 to allow for better re-ranking
            res = self.index.query(
                vector=query_vector,
                top_k=40,
                include_metadata=True
            )
            
            # 3. Format matches
            raw_results = []
            for match in res['matches']:
                doc = match['metadata']
                # Calculate scores for your RankingService
                pop_score = self.ranking.compute_popularity(int(doc.get('stars', 0)), int(doc.get('downloads', 0)))
                
                final_score = self.ranking.compute_score(
                    semantic_relevance=match['score'],
                    popularity=pop_score,
                    optimization=0.5, # Default since we flattened metadata
                    freshness=0.5
                )
                
                raw_results.append({**doc, "score": final_score})

            # 4. Sort by the calculated score
            raw_results.sort(key=lambda x: x['score'], reverse=True)
            
            return {
                "query": query,
                "results": raw_results[:limit],
                "total": len(raw_results)
            }
        except Exception as e:
            logger.error(f"Pinecone Search Error: {e}")
            return {"query": query, "results": [], "total": 0}
