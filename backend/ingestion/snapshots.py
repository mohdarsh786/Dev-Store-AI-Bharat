"""S3 snapshot persistence for raw crawler payloads."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from typing import Any, Optional
from uuid import UUID

import boto3

from config import settings

logger = logging.getLogger(__name__)


class SnapshotStore:
    """Persists raw payload snapshots for replay and debugging."""

    def __init__(self, bucket_name: Optional[str] = None, prefix: Optional[str] = None, s3_client: Any = None):
        self.bucket_name = bucket_name or settings.s3_bucket_crawler_data
        self.prefix = prefix or settings.crawler_snapshot_prefix
        self.s3_client = s3_client or boto3.client("s3", region_name=settings.aws_region)

    def persist(self, source: str, run_id: UUID, raw_payload: Any, timestamp: Optional[datetime] = None) -> str:
        timestamp = timestamp or datetime.utcnow()
        key = (
            f"{self.prefix}/{source}/"
            f"{timestamp.strftime('%Y/%m/%d/%H%M%S')}/"
            f"{run_id}.json"
        )
        body = json.dumps(raw_payload, default=str).encode("utf-8")
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        logger.info("Persisted ingestion snapshot", extra={"source": source, "key": key})
        return key

    @staticmethod
    def retention_guidance() -> str:
        return (
            "Configure an S3 lifecycle rule on the crawler data bucket to transition raw snapshots "
            "to Glacier after 30 days and expire them after 90-180 days, depending on replay needs."
        )
