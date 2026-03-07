"""
OpenSearch Indexing Service

Handles indexing resources into OpenSearch
"""
from typing import List, Dict, Any, Optional


class IndexingService:
    """
    Service for indexing resources in OpenSearch
    
    Features:
    - Bulk indexing
    - Document ID alignment with Aurora
    - Retry handling
    """
    
    def __init__(self, opensearch_client):
        """
        Initialize indexing service
        
        Args:
            opensearch_client: OpenSearch client (from clients/opensearch.py)
        """
        self.opensearch = opensearch_client
        self.index_name = "devstore_resources"
    
    def index_resource(self, resource: Dict[str, Any]) -> bool:
        """
        Index a single resource
        
        Args:
            resource: Resource data with embedding
            
        Returns:
            True if successful
        """
        try:
            # Convert resource to OpenSearch document
            document = self._resource_to_document(resource)
            
            # Index document (use resource ID as document ID)
            self.opensearch.index(
                index=self.index_name,
                id=resource['id'],
                body=document
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to index resource {resource.get('id')}: {e}")
            return False
    
    def index_batch(
        self,
        resources: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Index multiple resources in bulk
        
        Args:
            resources: List of resources with embeddings
            batch_size: Bulk batch size
            
        Returns:
            Statistics: indexed, failed counts
        """
        stats = {
            'indexed': 0,
            'failed': 0,
        }
        
        for i in range(0, len(resources), batch_size):
            batch = resources[i:i + batch_size]
            
            # Prepare bulk operations
            bulk_body = []
            for resource in batch:
                # Index action
                bulk_body.append({
                    'index': {
                        '_index': self.index_name,
                        '_id': resource['id']
                    }
                })
                # Document
                bulk_body.append(self._resource_to_document(resource))
            
            # Execute bulk operation
            try:
                response = self.opensearch.bulk(body=bulk_body)
                
                # Count successes and failures
                for item in response.get('items', []):
                    if item.get('index', {}).get('status') in [200, 201]:
                        stats['indexed'] += 1
                    else:
                        stats['failed'] += 1
                        
            except Exception as e:
                print(f"Bulk indexing failed: {e}")
                stats['failed'] += len(batch)
        
        return stats
    
    def _resource_to_document(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert resource to OpenSearch document
        
        Maps Aurora fields to OpenSearch fields
        """
        document = {
            'name': resource.get('name', ''),
            'description': resource.get('description', ''),
            'source': resource.get('source', ''),
            'source_url': resource.get('source_url', ''),
            'author': resource.get('author', ''),
            'stars': resource.get('stars', 0),
            'downloads': resource.get('downloads', 0),
            'license': resource.get('license', 'Unknown'),
            'tags': resource.get('tags', []),
            'category': resource.get('category', 'solution'),
            'thumbnail_url': resource.get('thumbnail_url'),
            'readme_url': resource.get('readme_url'),
            'created_at': resource.get('created_at'),
            'updated_at': resource.get('updated_at'),
        }
        
        # Add embedding vector if present
        if resource.get('embedding_vector'):
            document['embedding_vector'] = resource['embedding_vector']
        
        # Add ranking scores if present
        if resource.get('rank_score'):
            document['rank_score'] = resource['rank_score']
        if resource.get('trending_score'):
            document['trending_score'] = resource['trending_score']
        
        return document
    
    def delete_resource(self, resource_id: str) -> bool:
        """Delete a resource from index"""
        try:
            self.opensearch.delete(
                index=self.index_name,
                id=resource_id
            )
            return True
        except Exception as e:
            print(f"Failed to delete resource {resource_id}: {e}")
            return False
    
    def refresh_index(self):
        """Refresh index to make changes visible"""
        try:
            self.opensearch.indices.refresh(index=self.index_name)
        except Exception as e:
            print(f"Failed to refresh index: {e}")
