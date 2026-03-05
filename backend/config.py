"""
Configuration management for DevStore backend
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    database_url: str
    db_pool_size: int = 20
    db_max_overflow: int = 10
    
    # AWS
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    
    # Redis (ElastiCache)
    redis_host: str
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_pool_size: int = 50
    redis_socket_timeout: int = 5
    
    # OpenSearch
    opensearch_host: str
    opensearch_port: int = 443
    opensearch_use_ssl: bool = True
    opensearch_index_name: str = "devstore_resources"
    
    # Bedrock
    bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v1"
    
    # S3
    s3_bucket_boilerplate: str
    s3_bucket_crawler_data: str
    
    # API
    api_rate_limit: int = 100
    api_timeout: int = 30
    cors_origins: list = ["*"]
    
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


# Global settings instance
settings = Settings()
