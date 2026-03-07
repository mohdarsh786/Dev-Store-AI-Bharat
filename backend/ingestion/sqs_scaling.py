"""Optional SQS fanout path for future ingestion scaling."""

from __future__ import annotations

import json
from typing import Iterable, Optional
from uuid import UUID

import boto3

from config import settings


class SQSIngestionFanout:
    """Thin producer/worker abstraction compatible with single-run mode."""

    def __init__(self, queue_url: Optional[str] = None, sqs_client=None):
        self.queue_url = queue_url or settings.ingestion_sqs_queue_url
        self.sqs_client = sqs_client or boto3.client("sqs", region_name=settings.aws_region)

    def enqueue_sources(self, run_id: UUID, sources: Iterable[str]) -> int:
        if not self.queue_url:
            return 0
        count = 0
        for source in sources:
            self.sqs_client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps({"run_id": str(run_id), "source": source}),
            )
            count += 1
        return count

    @staticmethod
    def parse_message(body: str) -> dict:
        payload = json.loads(body)
        return {"run_id": payload["run_id"], "source": payload["source"]}
