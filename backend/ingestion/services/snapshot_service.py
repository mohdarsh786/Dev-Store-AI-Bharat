"""
Snapshot Service

Stores raw source payloads in S3 for replay and debugging
"""
import json
import gzip
from datetime import datetime
from typing import Dict, List, Any, Optional


class SnapshotService:
    """
    Service for persisting raw source data to S3
    
    Features:
    - Compressed JSON storage
    - Organized by source and timestamp
    - Retention lifecycle support
    """
    
    def __init__(self, s3_client, bucket_name: str):
        """
        Initialize snapshot service
        
        Args:
            s3_client: S3 client (boto3)
            bucket_name: S3 bucket for snapshots
        """
        self.s3 = s3_client
        self.bucket = bucket_name
    
    def save_snapshot(
        self,
        source: str,
        data: List[Dict[str, Any]],
        run_id: str,
        timestamp: datetime = None
    ) -> str:
        """
        Save raw source data to S3
        
        Args:
            source: Source name (huggingface, openrouter, etc.)
            data: Raw data from source
            run_id: Ingestion run ID
            timestamp: Snapshot timestamp (default: now)
            
        Returns:
            S3 key of saved snapshot
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Generate S3 key
        # Format: snapshots/{source}/{year}/{month}/{day}/{run_id}.json.gz
        key = self._generate_key(source, timestamp, run_id)
        
        # Prepare snapshot data
        snapshot = {
            'source': source,
            'run_id': run_id,
            'timestamp': timestamp.isoformat(),
            'count': len(data),
            'data': data,
        }
        
        # Compress and upload
        try:
            # Convert to JSON
            json_data = json.dumps(snapshot, indent=2)
            
            # Compress
            compressed = gzip.compress(json_data.encode('utf-8'))
            
            # Upload to S3
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=compressed,
                ContentType='application/json',
                ContentEncoding='gzip',
                Metadata={
                    'source': source,
                    'run_id': run_id,
                    'count': str(len(data)),
                }
            )
            
            return key
            
        except Exception as e:
            print(f"Failed to save snapshot: {e}")
            raise
    
    def _generate_key(
        self,
        source: str,
        timestamp: datetime,
        run_id: str
    ) -> str:
        """Generate S3 key for snapshot"""
        return (
            f"snapshots/{source}/"
            f"{timestamp.year:04d}/"
            f"{timestamp.month:02d}/"
            f"{timestamp.day:02d}/"
            f"{run_id}.json.gz"
        )
    
    def load_snapshot(self, key: str) -> Dict[str, Any]:
        """
        Load snapshot from S3
        
        Args:
            key: S3 key
            
        Returns:
            Snapshot data
        """
        try:
            # Download from S3
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=key
            )
            
            # Decompress
            compressed = response['Body'].read()
            json_data = gzip.decompress(compressed).decode('utf-8')
            
            # Parse JSON
            snapshot = json.loads(json_data)
            
            return snapshot
            
        except Exception as e:
            print(f"Failed to load snapshot: {e}")
            raise
    
    def list_snapshots(
        self,
        source: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[str]:
        """
        List available snapshots
        
        Args:
            source: Filter by source
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of S3 keys
        """
        prefix = "snapshots/"
        if source:
            prefix += f"{source}/"
        
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            keys = [obj['Key'] for obj in response.get('Contents', [])]
            
            # Filter by date if specified
            if start_date or end_date:
                keys = self._filter_by_date(keys, start_date, end_date)
            
            return keys
            
        except Exception as e:
            print(f"Failed to list snapshots: {e}")
            return []
    
    def _filter_by_date(
        self,
        keys: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[str]:
        """Filter keys by date range"""
        filtered = []
        
        for key in keys:
            # Extract date from key
            # Format: snapshots/{source}/{year}/{month}/{day}/{run_id}.json.gz
            parts = key.split('/')
            if len(parts) >= 5:
                try:
                    year = int(parts[2])
                    month = int(parts[3])
                    day = int(parts[4])
                    key_date = datetime(year, month, day)
                    
                    if start_date and key_date < start_date:
                        continue
                    if end_date and key_date > end_date:
                        continue
                    
                    filtered.append(key)
                except (ValueError, IndexError):
                    continue
        
        return filtered
