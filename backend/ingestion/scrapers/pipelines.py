"""
Scrapy pipelines for resource processing
"""
import hashlib
import json
import logging
from typing import Dict, Any
import redis
import boto3
from scrapy.exceptions import DropItem
from config import settings, ingestion_settings

logger = logging.getLogger(__name__)


class DeduplicationPipeline:
    """
    Pipeline for deduplicating resources using Redis.
    
    Uses SHA256 hash of (source + name + source_url) to identify duplicates.
    """
    
    def __init__(self):
        self.redis_client = None
        self.hash_set_key = ingestion_settings.dedup_hash_set_key
        self.stats = {
            'new': 0,
            'duplicate': 0
        }
    
    def open_spider(self, spider):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=False,  # We're storing binary hashes
                socket_timeout=5
            )
            self.redis_client.ping()
            logger.info("DeduplicationPipeline: Connected to Redis")
        except Exception as e:
            logger.error(f"DeduplicationPipeline: Failed to connect to Redis: {e}")
            raise
    
    def close_spider(self, spider):
        """Log statistics and close Redis connection"""
        logger.info(
            f"DeduplicationPipeline: Processed {self.stats['new']} new resources, "
            f"skipped {self.stats['duplicate']} duplicates"
        )
        if self.redis_client:
            self.redis_client.close()
    
    def process_item(self, item: Dict[str, Any], spider):
        """
        Check if resource is duplicate and drop if so.
        
        Args:
            item: Resource item
            spider: Spider instance
            
        Returns:
            Item if new, raises DropItem if duplicate
        """
        try:
            # Generate hash
            resource_hash = self._generate_hash(item)
            
            # Check if hash exists in Redis set
            result = self.redis_client.sadd(self.hash_set_key, resource_hash)
            
            if result == 1:
                # New resource
                self.stats['new'] += 1
                logger.debug(f"New resource: {item['name']} from {item['source']}")
                return item
            else:
                # Duplicate resource
                self.stats['duplicate'] += 1
                logger.debug(f"Duplicate resource: {item['name']} from {item['source']}")
                raise DropItem(f"Duplicate resource: {item['source_url']}")
                
        except DropItem:
            raise
        except Exception as e:
            logger.error(f"DeduplicationPipeline error: {e}")
            # On error, allow item to pass through
            return item
    
    def _generate_hash(self, item: Dict[str, Any]) -> str:
        """
        Generate SHA256 hash for resource.
        
        Args:
            item: Resource item
            
        Returns:
            SHA256 hash string
        """
        hash_input = f"{item['source']}:{item['name']}:{item['source_url']}"
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()


class ValidationPipeline:
    """
    Pipeline for validating resource data.
    
    Ensures all required fields are present and valid.
    """
    
    REQUIRED_FIELDS = [
        'name',
        'description',
        'source',
        'source_url',
        'category',
    ]
    
    def __init__(self):
        self.stats = {
            'valid': 0,
            'invalid': 0
        }
    
    def close_spider(self, spider):
        """Log statistics"""
        logger.info(
            f"ValidationPipeline: {self.stats['valid']} valid resources, "
            f"{self.stats['invalid']} invalid resources"
        )
    
    def process_item(self, item: Dict[str, Any], spider):
        """
        Validate resource data.
        
        Args:
            item: Resource item
            spider: Spider instance
            
        Returns:
            Item if valid, raises DropItem if invalid
        """
        try:
            # Check required fields
            for field in self.REQUIRED_FIELDS:
                if field not in item or not item[field]:
                    raise DropItem(f"Missing required field: {field}")
            
            # Validate source_url format
            if not item['source_url'].startswith(('http://', 'https://')):
                raise DropItem(f"Invalid source_url: {item['source_url']}")
            
            # Validate category
            valid_categories = ['api', 'model', 'dataset', 'solution']
            if item['category'] not in valid_categories:
                raise DropItem(f"Invalid category: {item['category']}")
            
            # Ensure tags is a list
            if 'tags' in item and not isinstance(item['tags'], list):
                item['tags'] = [item['tags']]
            
            # Ensure numeric fields are valid
            for field in ['stars', 'downloads']:
                if field in item:
                    try:
                        item[field] = int(item[field])
                        if item[field] < 0:
                            item[field] = 0
                    except (ValueError, TypeError):
                        item[field] = 0
            
            self.stats['valid'] += 1
            return item
            
        except DropItem as e:
            self.stats['invalid'] += 1
            logger.warning(f"ValidationPipeline: {e}")
            raise


class SQSPipeline:
    """
    Pipeline for sending validated resources to SQS queue.
    """
    
    def __init__(self):
        self.sqs_client = None
        self.queue_url = ingestion_settings.sqs_queue_url
        self.batch = []
        self.batch_size = ingestion_settings.sqs_batch_size
        self.stats = {
            'sent': 0,
            'failed': 0
        }
    
    def open_spider(self, spider):
        """Initialize SQS client"""
        try:
            self.sqs_client = boto3.client(
                'sqs',
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
            logger.info("SQSPipeline: Initialized SQS client")
        except Exception as e:
            logger.error(f"SQSPipeline: Failed to initialize SQS client: {e}")
            raise
    
    def close_spider(self, spider):
        """Send remaining items in batch and log statistics"""
        if self.batch:
            self._send_batch()
        
        logger.info(
            f"SQSPipeline: Sent {self.stats['sent']} resources to SQS, "
            f"{self.stats['failed']} failed"
        )
    
    def process_item(self, item: Dict[str, Any], spider):
        """
        Add item to batch and send when batch is full.
        
        Args:
            item: Resource item
            spider: Spider instance
            
        Returns:
            Item
        """
        self.batch.append(item)
        
        if len(self.batch) >= self.batch_size:
            self._send_batch()
        
        return item
    
    def _send_batch(self):
        """Send batch of items to SQS"""
        if not self.batch:
            return
        
        try:
            # Prepare batch entries
            entries = []
            for i, item in enumerate(self.batch):
                entries.append({
                    'Id': str(i),
                    'MessageBody': json.dumps(item, default=str),
                    'MessageAttributes': {
                        'source': {
                            'StringValue': item['source'],
                            'DataType': 'String'
                        },
                        'category': {
                            'StringValue': item['category'],
                            'DataType': 'String'
                        }
                    }
                })
            
            # Send batch to SQS
            response = self.sqs_client.send_message_batch(
                QueueUrl=self.queue_url,
                Entries=entries
            )
            
            # Check for failures
            successful = len(response.get('Successful', []))
            failed = len(response.get('Failed', []))
            
            self.stats['sent'] += successful
            self.stats['failed'] += failed
            
            if failed > 0:
                logger.warning(f"SQSPipeline: {failed} messages failed to send")
                for failure in response.get('Failed', []):
                    logger.error(f"SQSPipeline: Failed message {failure['Id']}: {failure['Message']}")
            
            logger.info(f"SQSPipeline: Sent batch of {successful} messages to SQS")
            
            # Clear batch
            self.batch = []
            
        except Exception as e:
            logger.error(f"SQSPipeline: Error sending batch to SQS: {e}")
            self.stats['failed'] += len(self.batch)
            self.batch = []
