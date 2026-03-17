"""
Configuration management for DevStore backend
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    neon_database_url: Optional[str] = None
    database_url: Optional[str] = None  # Legacy fallback support
    db_pool_size: int = 20
    db_max_overflow: int = 10
    
    # AWS
    aws_region: str = "us-east-1"
    aws_account_id: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    
    # Pinecone
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: str = "devstore_resources"
    
    # Legacy OpenSearch (Kept as Optional for fallback scripts)
    opensearch_host: Optional[str] = None
    opensearch_port: int = 443
    opensearch_use_ssl: bool = True
    opensearch_index_name: str = "devstore_resources"
    
    # Bedrock
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_claude_arn: Optional[str] = None
    
    # S3 (Legacy)
    s3_bucket_boilerplate: Optional[str] = None
    s3_bucket_crawler_data: Optional[str] = None

    # Ingestion
    ingestion_github_api_token: Optional[str] = None
    ingestion_sqs_queue_url: Optional[str] = None
    ingestion_lock_key: str = "ingestion:lock"
    ingestion_lock_ttl_seconds: int = 7200
    ingestion_status_ttl_seconds: int = 86400
    ingestion_schedule_hours: int = 12
    ranking_schedule_hour_utc: int = 3
    crawler_snapshot_prefix: str = "snapshots"

    # Additional System Parameters
    system_prompt: Optional[str] = None
    pinecone_environment: Optional[str] = None
    bedrock_region: Optional[str] = None

    # API
    api_rate_limit: int = 100
    api_timeout: int = 30
    cors_origins: list = ["*"]
    
    # Security
    secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    
    # Environment
    environment: str = "development"
    log_level: str = "INFO"
    
    # EC2 Deployment
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    workers: int = 4
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_database_url() -> str:
    """Compatibility helper for scripts that expect a function export."""
    return settings.database_url
