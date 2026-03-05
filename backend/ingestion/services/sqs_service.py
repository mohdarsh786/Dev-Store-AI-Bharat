"""
SQS Service

Handles SQS queue operations
"""
import json
import logging
from typing import List, Dict, Any
import boto3
from botocore.exceptions import ClientError
from config import settings, ingestion_settings

logger = logging.getLogger(__name__)


class SQSService:
    """
    Service for interacting with Amazon SQS.
    
    Features:
    - Receive messages with long polling
    - Parse and validate messages
    - Delete processed messages
    - Batch operations
    """
    
    def __init__(self):
        self.sqs_client = None
        self.queue_url = ingestion_settings.sqs_queue_url
        self.max_messages = ingestion_settings.sqs_max_messages
        self.wait_time = ingestion_settings.sqs_wait_time_seconds
        self.visibility_timeout = ingestion_settings.sqs_visibility_timeout
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize SQS client"""
        try:
            self.sqs_client = boto3.client(
                'sqs',
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
            logger.info("Initialized SQS client")
        except Exception as e:
            logger.error(f"Failed to initialize SQS client: {e}")
            raise
    
    async def receive_messages(
        self,
        max_messages: int = None
    ) -> List[Dict[str, Any]]:
        """
        Receive messages from SQS queue.
        
        Args:
            max_messages: Maximum number of messages to receive
            
        Returns:
            List of messages
        """
        max_messages = max_messages or self.max_messages
        
        try:
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=min(max_messages, 10),  # SQS limit is 10
                WaitTimeSeconds=self.wait_time,
                VisibilityTimeout=self.visibility_timeout,
                MessageAttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            logger.info(f"Received {len(messages)} messages from SQS")
            return messages
            
        except ClientError as e:
            logger.error(f"Error receiving messages from SQS: {e}")
            return []
    
    def parse_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse SQS message and extract resource data.
        
        Args:
            message: SQS message
            
        Returns:
            Resource data
        """
        try:
            body = json.loads(message['Body'])
            
            # Add message metadata
            body['_message_id'] = message['MessageId']
            body['_receipt_handle'] = message['ReceiptHandle']
            
            return body
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing message body: {e}")
            raise
    
    async def delete_message(self, receipt_handle: str) -> bool:
        """
        Delete a message from the queue.
        
        Args:
            receipt_handle: Message receipt handle
            
        Returns:
            True if successful
        """
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            return True
        except ClientError as e:
            logger.error(f"Error deleting message: {e}")
            return False
    
    async def delete_messages(self, receipt_handles: List[str]) -> int:
        """
        Delete multiple messages from the queue.
        
        Args:
            receipt_handles: List of receipt handles
            
        Returns:
            Number of messages deleted
        """
        if not receipt_handles:
            return 0
        
        deleted_count = 0
        
        # Process in batches of 10 (SQS limit)
        for i in range(0, len(receipt_handles), 10):
            batch = receipt_handles[i:i+10]
            
            try:
                entries = [
                    {'Id': str(j), 'ReceiptHandle': handle}
                    for j, handle in enumerate(batch)
                ]
                
                response = self.sqs_client.delete_message_batch(
                    QueueUrl=self.queue_url,
                    Entries=entries
                )
                
                deleted_count += len(response.get('Successful', []))
                
                if response.get('Failed'):
                    logger.warning(f"Failed to delete {len(response['Failed'])} messages")
                
            except ClientError as e:
                logger.error(f"Error deleting message batch: {e}")
        
        logger.info(f"Deleted {deleted_count} messages from SQS")
        return deleted_count
    
    async def get_queue_attributes(self) -> Dict[str, Any]:
        """
        Get queue attributes.
        
        Returns:
            Queue attributes
        """
        try:
            response = self.sqs_client.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=['All']
            )
            return response.get('Attributes', {})
        except ClientError as e:
            logger.error(f"Error getting queue attributes: {e}")
            return {}
