"""
Script to run RAG data ingestion.

Usage:
    python run_rag_ingestion.py
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run data ingestion pipeline"""
    try:
        # Import dependencies
        from clients.bedrock import BedrockClient
        from clients.opensearch import OpenSearchClient
        from rag.ingestor import DataIngestor
        
        logger.info("=" * 60)
        logger.info("RAG DATA INGESTION PIPELINE")
        logger.info("=" * 60)
        
        # Initialize clients
        logger.info("Initializing clients...")
        bedrock_client = BedrockClient()
        opensearch_client = OpenSearchClient()
        
        # Check OpenSearch connection
        health = opensearch_client.health_check()
        if health['status'] != 'healthy':
            logger.error(f"OpenSearch unhealthy: {health}")
            return 1
        
        logger.info(f"✅ OpenSearch connected: {health['cluster_name']}")
        
        # Ensure index exists
        if not opensearch_client.index_exists():
            logger.info("Creating k-NN index...")
            opensearch_client.create_knn_index(
                vector_dimension=1024,
                vector_field="embedding"
            )
            logger.info("✅ Index created")
        else:
            logger.info("✅ Index already exists")
        
        # Initialize ingestor
        logger.info("Initializing data ingestor...")
        ingestor = DataIngestor(
            bedrock_client=bedrock_client,
            opensearch_client=opensearch_client,
            chunk_size=500,
            batch_size=10
        )
        
        # Run ingestion
        logger.info("\nStarting ingestion...")
        logger.info("-" * 60)
        
        stats = ingestor.ingest_all(data_dir=Path("backend"))
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Files processed: {stats['files_processed']}")
        logger.info(f"Total records: {stats['total_records']}")
        logger.info(f"Successfully indexed: {stats['total_indexed']}")
        logger.info(f"Failed: {stats['total_failed']}")
        logger.info(f"Success rate: {stats['total_indexed']/stats['total_records']*100:.1f}%")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
