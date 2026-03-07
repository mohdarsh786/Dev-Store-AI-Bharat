"""
Resource Repository

Handles database operations for resources with upsert logic
"""
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class ResourceRepository:
    """
    Repository for resource CRUD operations
    
    Implements:
    - Upsert (insert or update) based on source + source_url
    - Deduplication using canonical key
    - Change detection via content hash
    """
    
    def __init__(self, db_client):
        """
        Initialize repository
        
        Args:
            db_client: Database client (from clients/database.py)
        """
        self.db = db_client
    
    def upsert_resource(self, resource: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Insert or update a resource
        
        Args:
            resource: Normalized resource data
            
        Returns:
            Tuple of (resource_id, is_new)
            - resource_id: UUID of the resource
            - is_new: True if inserted, False if updated
        """
        # Generate canonical key
        canonical_key = self._generate_canonical_key(
            resource['source'],
            resource['source_url']
        )
        
        # Generate content hash for change detection
        content_hash = self._generate_content_hash(resource)
        
        # Check if resource exists
        existing = self._find_by_canonical_key(canonical_key)
        
        if existing:
            # Check if content changed
            if existing['content_hash'] == content_hash:
                # No changes, skip update
                return existing['id'], False
            
            # Update existing resource
            resource_id = self._update_resource(existing['id'], resource, content_hash)
            return resource_id, False
        else:
            # Insert new resource
            resource_id = self._insert_resource(resource, canonical_key, content_hash)
            return resource_id, True
    
    def upsert_batch(self, resources: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Upsert multiple resources in batch
        
        Returns:
            Statistics: inserted, updated, skipped counts
        """
        stats = {
            'inserted': 0,
            'updated': 0,
            'skipped': 0,
            'failed': 0,
        }
        
        for resource in resources:
            try:
                resource_id, is_new = self.upsert_resource(resource)
                if is_new:
                    stats['inserted'] += 1
                else:
                    stats['updated'] += 1
            except Exception as e:
                stats['failed'] += 1
                print(f"Failed to upsert resource: {e}")
        
        return stats
    
    def _generate_canonical_key(self, source: str, source_url: str) -> str:
        """Generate unique key from source + source_url"""
        return f"{source}:{source_url}"
    
    def _generate_content_hash(self, resource: Dict[str, Any]) -> str:
        """
        Generate hash of resource content for change detection
        
        Includes: name, description, tags, metadata
        Excludes: timestamps, scraped_at
        """
        content = {
            'name': resource.get('name'),
            'description': resource.get('description'),
            'tags': sorted(resource.get('tags', [])),
            'stars': resource.get('stars'),
            'downloads': resource.get('downloads'),
            'license': resource.get('license'),
            'metadata': resource.get('metadata'),
        }
        
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _find_by_canonical_key(self, canonical_key: str) -> Optional[Dict[str, Any]]:
        """Find existing resource by canonical key"""
        query = """
            SELECT id, content_hash, name, source, source_url
            FROM resources
            WHERE canonical_key = %s
            LIMIT 1
        """
        
        result = self.db.fetch_one(query, (canonical_key,))
        return result
    
    def _insert_resource(
        self,
        resource: Dict[str, Any],
        canonical_key: str,
        content_hash: str
    ) -> str:
        """Insert new resource"""
        query = """
            INSERT INTO resources (
                canonical_key,
                content_hash,
                name,
                description,
                source,
                source_url,
                author,
                stars,
                downloads,
                license,
                tags,
                version,
                category,
                thumbnail_url,
                readme_url,
                metadata,
                created_at,
                updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, NOW(), NOW()
            )
            RETURNING id
        """
        
        params = (
            canonical_key,
            content_hash,
            resource['name'],
            resource.get('description', ''),
            resource['source'],
            resource['source_url'],
            resource.get('author', ''),
            resource.get('stars', 0),
            resource.get('downloads', 0),
            resource.get('license', 'Unknown'),
            json.dumps(resource.get('tags', [])),
            resource.get('version', ''),
            resource.get('category', 'solution'),
            resource.get('thumbnail_url'),
            resource.get('readme_url'),
            json.dumps(resource.get('metadata', {})),
        )
        
        result = self.db.execute(query, params)
        return result['id']
    
    def _update_resource(
        self,
        resource_id: str,
        resource: Dict[str, Any],
        content_hash: str
    ) -> str:
        """Update existing resource"""
        query = """
            UPDATE resources
            SET
                content_hash = %s,
                name = %s,
                description = %s,
                author = %s,
                stars = %s,
                downloads = %s,
                license = %s,
                tags = %s,
                version = %s,
                category = %s,
                thumbnail_url = %s,
                readme_url = %s,
                metadata = %s,
                updated_at = NOW()
            WHERE id = %s
            RETURNING id
        """
        
        params = (
            content_hash,
            resource['name'],
            resource.get('description', ''),
            resource.get('author', ''),
            resource.get('stars', 0),
            resource.get('downloads', 0),
            resource.get('license', 'Unknown'),
            json.dumps(resource.get('tags', [])),
            resource.get('version', ''),
            resource.get('category', 'solution'),
            resource.get('thumbnail_url'),
            resource.get('readme_url'),
            json.dumps(resource.get('metadata', {})),
            resource_id,
        )
        
        result = self.db.execute(query, params)
        return result['id']
    
    def get_resources_needing_embeddings(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get resources that need embeddings generated
        
        Returns resources where:
        - embedding_vector IS NULL
        - OR content_hash changed since last embedding
        """
        query = """
            SELECT id, name, description, tags, category, content_hash
            FROM resources
            WHERE embedding_vector IS NULL
               OR embedding_content_hash != content_hash
            LIMIT %s
        """
        
        return self.db.fetch_all(query, (limit,))
    
    def update_embedding(
        self,
        resource_id: str,
        embedding_vector: List[float],
        content_hash: str
    ):
        """Update resource with embedding vector"""
        query = """
            UPDATE resources
            SET
                embedding_vector = %s,
                embedding_content_hash = %s,
                embedding_generated_at = NOW()
            WHERE id = %s
        """
        
        # Convert embedding to PostgreSQL array format
        embedding_str = '{' + ','.join(map(str, embedding_vector)) + '}'
        
        self.db.execute(query, (embedding_str, content_hash, resource_id))
