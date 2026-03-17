import os
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from clients.bedrock import BedrockClient
from services.ranking import RankingService
from clients.database import DatabaseClient

import threading
import time

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self):
        self.bedrock = BedrockClient()
        self.ranking = RankingService()
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        self.db = None # Lazy init
        self._init_db()

    def _init_db(self):
        try:
            self.db = DatabaseClient()
        except Exception as e:
            logger.warning(f"Failed to init DB in SearchService: {e}")
            self.db = None

    def _trigger_background_sync(self, resources: List[Dict[str, Any]]):
        """Non-blocking background sync for items found in SQL but potentially missing in Pinecone."""
        def sync_worker():
            try:
                # Basic rate limiting and avoidance of duplicate work
                for res in resources[:5]: # Just sync top few missing ones per query
                    res_id = str(res.get('id'))
                    # Check if already indexed could be done here, but we'll just upsert to be sure
                    # (Pinecone upsert is idempotent with same ID)
                    text = f"{res.get('name')} {res.get('type')} {res.get('description')}"
                    vector = self.bedrock.generate_embedding(text)
                    if vector:
                        self.index.upsert(vectors=[{
                            "id": f"{res_id}-full",
                            "values": vector,
                            "metadata": {
                                "resource_id": res_id,
                                "name": res.get('name'),
                                "category": res.get('type') or 'api',
                                "text_content": res.get('description', '')[:2000]
                            }
                        }])
            except Exception as e:
                logger.error(f"Background Sync Error: {e}")

        threading.Thread(target=sync_worker, daemon=True).start()

    def _normalize_and_format(self, raw_results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        final_output = []
        if not raw_results:
            return final_output
            
        # 5. Sort by original score
        raw_results.sort(key=lambda x: x.get('score', 0), reverse=True)

        max_score = raw_results[0]['score']
        
        # Quality Gate: If the top result is garbage (total score < 0.4), ignore the batch
        if max_score < 0.4:
            return []
            
        if max_score <= 0: max_score = 1.0
        
        for r in raw_results[:limit]:
            # Batch scaling ensures variance while keeping scores under 100%
            display_score = round((r['score'] / max_score) * 0.99, 4)
            
            # Multi-layer Fallbacks
            category = r.get('category') or r.get('type')
            if not category or str(category).lower() == 'unknown':
                name_lower = str(r.get('name', '')).lower()
                if 'dataset' in name_lower: category = 'dataset'
                elif 'api' in name_lower: category = 'api'
                else: category = 'model'
            
            pricing = r.get('pricing_type') or 'free'
            is_free_val = r.get('is_free')
            if is_free_val is None:
                is_free = (str(pricing).lower() == 'free')
            else:
                is_free = str(is_free_val).lower() == 'true' if isinstance(is_free_val, str) else bool(is_free_val)
            
            final_output.append({
                **r,
                "score": display_score,
                "category": category,
                "pricing_type": pricing,
                "is_free": is_free,
                "insights": r.get('insights', r.get('description', '')[:200]),
                "code_snippet": r.get('code_snippet', ''),
                "stars": int(r.get('stars') or r.get('github_stars') or 0),
                "downloads": int(r.get('downloads') or r.get('download_count') or 0)
            })
        return final_output

    def search(self, query: str, limit: int = 20, resource_type: Optional[str] = None, pricing_type: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        try:
            raw_results = []
            seen_ids = set()

            # 0. Ensure DB is connected
            if not self.db:
                self._init_db()

            # 1. Exact / Keyword Match (Hybrid SQL Layer) with filtering
            if self.db:
                # Build WHERE clause with filters
                where_conditions = ["(name ILIKE %s OR type ILIKE %s)"]
                params = [f"%{query}%", f"%{query}%"]
                
                if resource_type:
                    where_conditions.append("type = %s")
                    params.append(resource_type.lower())
                    
                if pricing_type:
                    where_conditions.append("pricing_type = %s")
                    params.append(pricing_type.lower())
                
                where_clause = " AND ".join(where_conditions)
                sql_query = f"""
                    SELECT * FROM resources 
                    WHERE {where_clause}
                    LIMIT %s
                """
                params.append(limit)
                
                db_matches = self.db.execute_query(sql_query, tuple(params))
                if db_matches:
                    for doc in db_matches:
                        res_id = str(doc.get('id', ''))
                        if res_id not in seen_ids:
                            seen_ids.add(res_id)
                            # Boost SQL score
                            pop_score = self.ranking.compute_popularity(
                                int(doc.get('github_stars') or doc.get('stars') or 0), 
                                int(doc.get('downloads') or 0)
                            )
                            # Priority for exact keywords: artificial semantic relevance 1.0
                            final_score = self.ranking.compute_score(
                                semantic_relevance=1.0,
                                popularity=pop_score,
                                optimization=0.5,
                                freshness=0.5
                            )
                            
                            # Give a large absolute multiplier to push SQL matches to top
                            final_score += 2.0
                            
                            # Ensure category is set correctly from 'type' column
                            category = doc.get('type') or 'api'
                            if category.lower() == 'unknown':
                                category = 'api'
                                
                            formatted_doc = {
                                **doc,
                                "id": res_id,
                                "name": doc.get('name', ''),
                                "category": category,
                                "stars": int(doc.get('github_stars') or doc.get('stars') or 0),
                                "downloads": int(doc.get('downloads') or 0),
                                "score": final_score,
                                "text_content": doc.get('description', '')
                            }
                            raw_results.append(formatted_doc)

            # 2. Vectorize query (Timeout protected)
            try:
                import asyncio
                # Fast embedding generation
                query_vector = self.bedrock.generate_embedding(query)
            except Exception as e:
                logger.warning(f"Embedding generation failed, returning SQL matches only: {e}")
                results = self._normalize_and_format(raw_results, limit)
                return {
                    "query": query,
                    "results": results,
                    "total": len(raw_results),
                    "error": "embedding_failure",
                    "latency_optimized": True
                }
            
            # 3. Query Pinecone (Semantic Layer) with metadata filtering
            try:
                # Build Pinecone metadata filter
                pinecone_filter = {}
                if resource_type:
                    pinecone_filter["type"] = {"$eq": resource_type.lower()}
                if pricing_type:
                    # Convert pricing_type to is_free boolean for Pinecone
                    is_free = pricing_type.lower() == "free"
                    pinecone_filter["is_free"] = {"$eq": is_free}
                
                query_params = {
                    "vector": query_vector,
                    "top_k": 40,
                    "include_metadata": True,
                    "timeout": 5  # 5 second hard cap for vector search
                }
                
                if pinecone_filter:
                    query_params["filter"] = pinecone_filter
                
                res = self.index.query(**query_params)
            except Exception as e:
                logger.warning(f"Pinecone query timed out, falling back to SQL matches: {e}")
                res = {"matches": []}
            
            # 4. Format Pinecone matches
            for match in res['matches']:
                doc = match['metadata']
                
                # Check for duplication using the prefix of the chunk ID
                raw_chunk_id = match['id']
                base_resource_id = str(doc.get('resource_id', raw_chunk_id.split('-chunk')[0]))
                
                # Also deduplicate by name just to be very safe
                res_name = doc.get('name', '')
                if base_resource_id not in seen_ids and not any(r.get('name') == res_name for r in raw_results):
                    seen_ids.add(base_resource_id)
                    # Calculate scores for your RankingService
                    pop_score = self.ranking.compute_popularity(int(doc.get('stars', 0)), int(doc.get('downloads', 0)))
                    
                    final_score = self.ranking.compute_score(
                        semantic_relevance=match['score'],
                        popularity=pop_score,
                        optimization=0.5, # Default since we flattened metadata
                        freshness=0.5
                    )
                    
                    raw_results.append({
                        **doc, 
                        "id": match['id'], 
                        "score": final_score
                    })

            # 6. Advanced Normalization: Batch-wide Min-Max Scaling [0.0 - 0.99]
            final_output = self._normalize_and_format(raw_results, limit)
            
            # Trigger background sync for SQL items to ensure they reach the vector store
            sql_only_items = [r for r in raw_results if "-chunk" not in str(r.get('id'))]
            if sql_only_items:
                self._trigger_background_sync(sql_only_items)
            
            return {
                "query": query,
                "results": final_output,
                "total": len(raw_results),
                "latency_optimized": True
            }
        except Exception as e:
            logger.error(f"Search Error: {e}")
            if "connection" in str(e).lower() or "closed" in str(e).lower():
                self.db = None # Trigger re-init next time
            # Fast-path fallback to SQL if everything fails
            return {"query": query, "results": [], "total": 0}

    def trending(self, resource_type: Optional[str] = None, pricing_type: Optional[str] = None, sort_by: Optional[str] = None, limit: int = 40) -> Dict[str, Any]:
        """Get trending resources with filtering support."""
        try:
            # 0. Ensure DB is connected
            if not self.db:
                self._init_db()

            raw_results = []
            
            if self.db:
                # Build WHERE clause with filters
                where_conditions = []
                params = []
                
                if resource_type:
                    where_conditions.append("type = %s")
                    params.append(resource_type.lower())
                    
                if pricing_type:
                    where_conditions.append("pricing_type = %s")
                    params.append(pricing_type.lower())
                
                where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
                
                # Build ORDER BY clause
                if sort_by == "downloads":
                    order_clause = "ORDER BY download_count DESC, github_stars DESC"
                elif sort_by == "popularity":
                    order_clause = "ORDER BY github_stars DESC, download_count DESC"
                elif sort_by == "paid":
                    order_clause = "ORDER BY github_stars DESC, rank_score DESC"
                else:
                    order_clause = "ORDER BY rank_score DESC, trending_score DESC, github_stars DESC"
                
                sql_query = f"""
                    SELECT * FROM resources
                    {where_clause}
                    {order_clause}
                    LIMIT %s
                """
                params.append(limit)
                
                db_matches = self.db.execute_query(sql_query, tuple(params))
                if db_matches:
                    for doc in db_matches:
                        res_id = str(doc.get('id', ''))
                        # Calculate popularity score
                        pop_score = self.ranking.compute_popularity(
                            int(doc.get('github_stars') or doc.get('stars') or 0), 
                            int(doc.get('downloads') or doc.get('download_count') or 0)
                        )
                        
                        # Use rank_score if available, otherwise compute
                        final_score = float(doc.get('rank_score') or pop_score)
                        
                        # Ensure category is set correctly from 'type' column
                        category = doc.get('type') or 'api'
                        if category.lower() == 'unknown':
                            category = 'api'
                            
                        formatted_doc = {
                            **doc,
                            "id": res_id,
                            "name": doc.get('name', ''),
                            "category": category,
                            "stars": int(doc.get('github_stars') or doc.get('stars') or 0),
                            "downloads": int(doc.get('downloads') or doc.get('download_count') or 0),
                            "score": final_score,
                            "text_content": doc.get('description', '')
                        }
                        raw_results.append(formatted_doc)

            # Apply normalization with Min-Max scaling within filtered results
            final_output = self._normalize_and_format(raw_results, limit)
            
            return {
                "query": "trending",
                "results": final_output,
                "total": len(raw_results),
                "source": "database"
            }
        except Exception as e:
            logger.error(f"Trending Error: {e}")
            if "connection" in str(e).lower() or "closed" in str(e).lower():
                self.db = None # Trigger re-init next time
            return {"query": "trending", "results": [], "total": 0}
