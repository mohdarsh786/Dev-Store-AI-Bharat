"""
Example script for indexing resources with embeddings

This script demonstrates how to:
1. Fetch resources from the database
2. Generate embeddings using Bedrock Titan v2
3. Index documents in OpenSearch with k-NN vectors

Usage:
    python index_resources_example.py
"""
import logging
from typing import List, Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def index_resources():
    """Index resources with embeddings in OpenSearch"""
    try:
        # Import services
        from services.embeddings import get_embeddings_service
        from clients.opensearch import OpenSearchClient
        from clients.database import DatabaseClient
        
        logger.info("Initializing services...")
        embeddings_service = get_embeddings_service(model_version="v2")
        opensearch_client = OpenSearchClient()
        db_client = DatabaseClient()
        
        # Verify OpenSearch index exists
        if not opensearch_client.index_exists():
            logger.error("OpenSearch index does not exist. Run setup_opensearch_index.py first.")
            return 1
        
        logger.info("Fetching resources from database...")
        # Get all resources from database
        # NOTE: Replace with your actual database query
        resources = db_client.get_all_resources()
        logger.info(f"Found {len(resources)} resources to index")
        
        # Index each resource
        indexed_count = 0
        failed_count = 0
        
        for i, resource in enumerate(resources):
            try:
                logger.info(f"Processing resource {i+1}/{len(resources)}: {resource.name}")
                
                # Generate embedding for resource
                embedding = embeddings_service.get_resource_embedding({
                    'name': resource.name,
                    'description': resource.description or '',
                    'tags': resource.tags or [],
                    'resource_type': resource.resource_type
                })
                
                # Prepare document for indexing
                document = {
                    'name': resource.name,
                    'description': resource.description or '',
                    'resource_type': resource.resource_type,
                    'pricing_type': resource.pricing_type or 'free',
                    'source': resource.source or 'github',
                    'github_stars': resource.github_stars or 0,
                    'downloads': resource.downloads or 0,
                    'last_updated': resource.last_updated.isoformat() if resource.last_updated else datetime.utcnow().isoformat(),
                    'health_status': resource.health_status or 'unknown',
                    'embedding': embedding  # 1024-dimensional vector
                }
                
                # Index document in OpenSearch
                response = opensearch_client.index_document(
                    document=document,
                    doc_id=str(resource.id),
                    refresh=False  # Batch refresh at the end
                )
                
                indexed_count += 1
                logger.debug(f"Indexed resource {resource.id}: {response['_id']}")
                
            except Exception as e:
                logger.error(f"Failed to index resource {resource.id}: {e}")
                failed_count += 1
                continue
        
        # Refresh index to make documents searchable
        logger.info("Refreshing OpenSearch index...")
        opensearch_client._client.indices.refresh(index=opensearch_client.index_name)
        
        logger.info(f"✅ Indexing complete!")
        logger.info(f"  - Successfully indexed: {indexed_count}")
        logger.info(f"  - Failed: {failed_count}")
        logger.info(f"  - Total: {len(resources)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error indexing resources: {e}", exc_info=True)
        return 1


def index_mock_resources():
    """Index mock resources for testing (when database is not available)"""
    try:
        from services.embeddings import get_embeddings_service
        from clients.opensearch import OpenSearchClient
        
        logger.info("Initializing services...")
        embeddings_service = get_embeddings_service(model_version="v2")
        opensearch_client = OpenSearchClient()
        
        # Mock resources for testing
        mock_resources = [
            {
                'id': '1',
                'name': 'OpenAI GPT-4 API',
                'description': 'Advanced language model API for natural language processing, text generation, and conversational AI',
                'resource_type': 'API',
                'pricing_type': 'paid',
                'source': 'github',
                'github_stars': 50000,
                'downloads': 1000000,
                'health_status': 'healthy'
            },
            {
                'id': '2',
                'name': 'Hugging Face Transformers',
                'description': 'State-of-the-art machine learning models for NLP tasks including classification, translation, and generation',
                'resource_type': 'Model',
                'pricing_type': 'free',
                'source': 'huggingface',
                'github_stars': 75000,
                'downloads': 2500000,
                'health_status': 'healthy'
            },
            {
                'id': '3',
                'name': 'Common Crawl Dataset',
                'description': 'Petabyte-scale web crawl data for training large language models and research',
                'resource_type': 'Dataset',
                'pricing_type': 'free',
                'source': 'kaggle',
                'github_stars': 1200,
                'downloads': 500000,
                'health_status': 'healthy'
            },
            {
                'id': '4',
                'name': 'Llama 3 70B',
                'description': 'Meta\'s flagship open-source LLM, fine-tuned for reasoning and instruction following',
                'resource_type': 'Model',
                'pricing_type': 'free',
                'source': 'huggingface',
                'github_stars': 45000,
                'downloads': 210000,
                'health_status': 'healthy'
            },
            {
                'id': '5',
                'name': 'Stable Diffusion XL',
                'description': 'State-of-the-art text-to-image generation model with high-quality outputs',
                'resource_type': 'Model',
                'pricing_type': 'free',
                'source': 'huggingface',
                'github_stars': 52000,
                'downloads': 980000,
                'health_status': 'healthy'
            }
        ]
        
        logger.info(f"Indexing {len(mock_resources)} mock resources...")
        
        for resource in mock_resources:
            # Generate embedding
            embedding = embeddings_service.get_resource_embedding({
                'name': resource['name'],
                'description': resource['description'],
                'tags': [],
                'resource_type': resource['resource_type']
            })
            
            # Prepare document
            document = {
                **resource,
                'last_updated': datetime.utcnow().isoformat(),
                'embedding': embedding
            }
            
            # Index document (refresh ignored for OpenSearch Serverless)
            opensearch_client.index_document(
                document=document,
                doc_id=resource['id'],
                refresh=False
            )
            
            logger.info(f"Indexed: {resource['name']}")
        
        logger.info("✅ Mock resources indexed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error indexing mock resources: {e}", exc_info=True)
        return 1


def main():
    """Main entry point"""
    import sys
    
    # Check if --mock flag is provided
    use_mock = '--mock' in sys.argv
    
    if use_mock:
        logger.info("Using mock resources for testing")
        return index_mock_resources()
    else:
        logger.info("Indexing resources from database")
        return index_resources()


if __name__ == "__main__":
    exit(main())
