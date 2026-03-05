"""
Ingestion Pipeline Configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional


class IngestionSettings(BaseSettings):
    """Ingestion pipeline settings"""
    
    # Scraper Configuration
    scraper_lock_key: str = "devstore:scraper_lock"
    scraper_lock_ttl: int = 1500  # 25 minutes
    scraper_batch_size: int = 100
    scraper_concurrent_requests: int = 16
    scraper_download_delay: float = 0.5
    
    # GitHub API
    github_api_token: Optional[str] = None
    github_api_base_url: str = "https://api.github.com"
    github_rate_limit: int = 5000
    
    # HuggingFace API
    huggingface_api_token: Optional[str] = None
    huggingface_api_base_url: str = "https://huggingface.co/api"
    
    # RapidAPI
    rapidapi_key: Optional[str] = None
    rapidapi_base_url: str = "https://rapidapi.com/api"
    
    # Deduplication
    dedup_hash_set_key: str = "devstore:unique_hashes"
    dedup_ttl: int = 86400 * 30  # 30 days
    
    # SQS Configuration
    sqs_queue_url: str
    sqs_batch_size: int = 10
    sqs_wait_time_seconds: int = 20
    sqs_visibility_timeout: int = 300
    sqs_max_messages: int = 10
    
    # Worker Configuration
    worker_batch_size: int = 20
    worker_poll_interval: int = 5
    worker_max_retries: int = 3
    worker_retry_delay: int = 5
    
    # Embedding Configuration
    embedding_model_id: str = "amazon.titan-embed-text-v1"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 25
    embedding_cache_ttl: int = 86400  # 24 hours
    
    # OpenSearch Configuration
    opensearch_index_name: str = "dev-store-resources"
    opensearch_bulk_size: int = 100
    opensearch_refresh_interval: str = "30s"
    
    # Monitoring
    cloudwatch_namespace: str = "DevStore/Ingestion"
    cloudwatch_log_group: str = "/aws/devstore/ingestion"
    
    # S3 Configuration
    s3_bucket_artifacts: str = "devstore-ingestion-artifacts"
    s3_readme_prefix: str = "readmes/"
    s3_metadata_prefix: str = "metadata/"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        env_prefix = "INGESTION_"


# Global settings instance
ingestion_settings = IngestionSettings()
