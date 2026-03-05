"""
Batch Processing Worker

Polls SQS queue, processes resources, generates embeddings, and stores in database and OpenSearch.
"""
import asyncio
import logging
import time
from typing import List, Dict, Any
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings, ingestion_settings
from workers.embedder import EmbeddingService
from services.sqs_service import SQSService
from services.storage_service import StorageService
from services.indexing_service import IndexingService
from monitoring.cloudwatch_metrics import CloudWatchMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Main batch processing worker.
    
    Workflow:
    1. Poll SQS for messages
    2. Parse and normalize resources
    3. Generate embeddings via Bedrock
    4. Store in Aurora PostgreSQL
    5. Index in OpenSearch
    6. Update Redis caches
    7. Delete messages from SQS
    """
    
    def __init__(self):
        self.sqs_service = SQSService()
        self.embedding_service = EmbeddingService()
        self.storage_service = StorageService()
        self.indexing_service = IndexingService()
        self.metrics = CloudWatchMetrics()
        
        self.running = False
        self.stats = {
            'batches_processed': 0,
            'resources_processed': 0,
            'resources_stored': 0,
            'resources_indexed': 0,
            'errors': 0
        }
    
    async def start(self):
        """Start the batch processor"""
        logger.info("Starting batch processor...")
        self.running = True
        
        try:
            # Initialize services
            await self._initialize_services()
            
            # Start processing loop
            await self._processing_loop()
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Fatal error in batch processor: {e}", exc_info=True)
        finally:
            await self._shutdown()
    
    async def _initialize_services(self):
        """Initialize all services"""
        logger.info("Initializing services...")
        
        # Services are initialized on first use
        # This method can be used for any pre-flight checks
        
        logger.info("Services initialized successfully")
    
    async def _processing_loop(self):
        """Main processing loop"""
        logger.info("Starting processing loop...")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Poll SQS for messages
                messages = await self.sqs_service.receive_messages(
                    max_messages=ingestion_settings.worker_batch_size
                )
                
                if not messages:
                    # No messages, wait before polling again
                    await asyncio.sleep(ingestion_settings.worker_poll_interval)
                    continue
                
                logger.info(f"Received {len(messages)} messages from SQS")
                
                # Process batch
                await self._process_batch(messages)
                
                # Record metrics
                batch_duration = time.time() - start_time
                await self.metrics.record_batch_processed(
                    batch_size=len(messages),
                    duration=batch_duration
                )
                
                self.stats['batches_processed'] += 1
                logger.info(f"Batch processed in {batch_duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                self.stats['errors'] += 1
                await asyncio.sleep(ingestion_settings.worker_retry_delay)
    
    async def _process_batch(self, messages: List[Dict[str, Any]]):
        """
        Process a batch of messages.
        
        Args:
            messages: List of SQS messages
        """
        resources = []
        receipt_handles = []
        
        # Parse messages
        for message in messages:
            try:
                resource = self.sqs_service.parse_message(message)
                resources.append(resource)
                receipt_handles.append(message['ReceiptHandle'])
            except Exception as e:
                logger.error(f"Error parsing message: {e}")
                self.stats['errors'] += 1
        
        if not resources:
            return
        
        # Process resources
        processed_resources = []
        for resource in resources:
            try:
                processed = await self._process_resource(resource)
                if processed:
                    processed_resources.append(processed)
                    self.stats['resources_processed'] += 1
            except Exception as e:
                logger.error(f"Error processing resource {resource.get('name')}: {e}")
                self.stats['errors'] += 1
        
        # Store in database
        stored_count = await self._store_resources(processed_resources)
        self.stats['resources_stored'] += stored_count
        
        # Index in OpenSearch
        indexed_count = await self._index_resources(processed_resources)
        self.stats['resources_indexed'] += indexed_count
        
        # Delete messages from SQS
        await self.sqs_service.delete_messages(receipt_handles)
        
        logger.info(
            f"Processed batch: {len(resources)} received, "
            f"{len(processed_resources)} processed, "
            f"{stored_count} stored, {indexed_count} indexed"
        )
    
    async def _process_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single resource.
        
        Args:
            resource: Raw resource data
            
        Returns:
            Processed resource with embedding
        """
        # Normalize metadata
        resource = self._normalize_resource(resource)
        
        # Generate embedding
        embedding_text = self._create_embedding_text(resource)
        embedding = await self.embedding_service.generate_embedding(embedding_text)
        
        if embedding:
            resource['embedding'] = embedding
        else:
            logger.warning(f"Failed to generate embedding for {resource['name']}")
        
        return resource
    
    def _normalize_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize resource data.
        
        Args:
            resource: Raw resource data
            
        Returns:
            Normalized resource
        """
        # Ensure required fields
        resource.setdefault('author', 'Unknown')
        resource.setdefault('license', 'Unknown')
        resource.setdefault('stars', 0)
        resource.setdefault('downloads', 0)
        resource.setdefault('tags', [])
        resource.setdefault('version', '1.0')
        resource.setdefault('metadata', {})
        
        # Limit tags to 10
        if len(resource['tags']) > 10:
            resource['tags'] = resource['tags'][:10]
        
        # Add timestamps
        resource['created_at'] = datetime.utcnow().isoformat()
        resource['updated_at'] = datetime.utcnow().isoformat()
        
        return resource
    
    def _create_embedding_text(self, resource: Dict[str, Any]) -> str:
        """
        Create text for embedding generation.
        
        Args:
            resource: Resource data
            
        Returns:
            Text string for embedding
        """
        parts = [
            resource['name'],
            resource['description'],
            ' '.join(resource.get('tags', []))
        ]
        return ' '.join(filter(None, parts))
    
    async def _store_resources(self, resources: List[Dict[str, Any]]) -> int:
        """
        Store resources in database.
        
        Args:
            resources: List of processed resources
            
        Returns:
            Number of resources stored
        """
        try:
            return await self.storage_service.store_resources(resources)
        except Exception as e:
            logger.error(f"Error storing resources: {e}", exc_info=True)
            return 0
    
    async def _index_resources(self, resources: List[Dict[str, Any]]) -> int:
        """
        Index resources in OpenSearch.
        
        Args:
            resources: List of processed resources
            
        Returns:
            Number of resources indexed
        """
        try:
            return await self.indexing_service.index_resources(resources)
        except Exception as e:
            logger.error(f"Error indexing resources: {e}", exc_info=True)
            return 0
    
    async def _shutdown(self):
        """Shutdown the processor"""
        logger.info("Shutting down batch processor...")
        self.running = False
        
        # Log final statistics
        logger.info(f"Final statistics: {self.stats}")
        
        # Close services
        await self.embedding_service.close()
        
        logger.info("Batch processor shutdown complete")


async def main():
    """Main entry point"""
    processor = BatchProcessor()
    await processor.start()


if __name__ == '__main__':
    asyncio.run(main())
