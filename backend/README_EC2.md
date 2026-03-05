# DevStore Backend - Centralized API for EC2

FastAPI backend for DevStore - AI-Powered Developer Marketplace with Redis caching and AWS integration.

## Architecture

```
FastAPI Application (api_gateway.py)
    ↓
├── Redis Cache (ElastiCache)
├── PostgreSQL (RDS Aurora)
├── OpenSearch (Vector Search)
└── Amazon Bedrock (AI/ML)
```

## Features

- **Centralized API Gateway**: All endpoints consolidated in one application
- **Redis Caching**: Multi-layer caching with ElastiCache integration
- **Semantic Search**: RAG-powered search with Bedrock and OpenSearch
- **Resource Management**: CRUD operations for APIs, Models, Datasets
- **Boilerplate Generation**: Code generation for Python/JS/TS
- **Health Monitoring**: Comprehensive health checks and metrics
- **Production Ready**: Gunicorn + Uvicorn with systemd integration

## Quick Start

### Local Development

1. **Create virtual environment:**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Run migrations:**
```bash
python run_migrations.py
```

5. **Start development server:**
```bash
python api_gateway.py
# Or with auto-reload:
uvicorn api_gateway:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment (EC2)

See [../EC2_DEPLOYMENT_GUIDE.md](../EC2_DEPLOYMENT_GUIDE.md) for complete deployment instructions.

**Quick deployment:**
```bash
# Make scripts executable
chmod +x start_server.sh deploy.sh

# Deploy
./deploy.sh
```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **OpenAPI JSON**: http://localhost:8000/api/openapi.json

## API Endpoints

### Health & Status
- `GET /api/v1/health` - Basic health check
- `GET /api/v1/health/detailed` - Detailed health with dependencies
- `GET /api/v1/metrics` - Application metrics
- `GET /api/v1/cache/stats` - Cache statistics

### Search
- `POST /api/v1/search` - Semantic search with filters

### Resources
- `GET /api/v1/resources` - List resources
- `GET /api/v1/resources/{id}` - Get resource details

### Categories
- `GET /api/v1/categories` - List categories
- `GET /api/v1/categories/{id}/resources` - Get category resources

### Boilerplate
- `POST /api/v1/boilerplate/generate` - Generate starter code

### Users
- `GET /api/v1/users/profile` - Get user profile
- `PUT /api/v1/users/profile` - Update profile
- `POST /api/v1/users/track` - Track user actions

## Configuration

### Environment Variables

**Required:**
```bash
DATABASE_URL=postgresql://user:pass@host:5432/devstore
REDIS_HOST=your-elasticache-endpoint.cache.amazonaws.com
OPENSEARCH_HOST=your-opensearch-endpoint.us-east-1.es.amazonaws.com
AWS_REGION=us-east-1
```

**Optional:**
```bash
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_POOL_SIZE=50
API_RATE_LIMIT=100
LOG_LEVEL=INFO
ENVIRONMENT=production
```

See `.env.ec2.example` for complete configuration.

## Project Structure

```
backend/
├── api_gateway.py          # Main application entry point
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── clients/                # External service clients
│   ├── redis_client.py     # Redis/ElastiCache client
│   ├── database.py         # PostgreSQL client
│   ├── opensearch.py       # OpenSearch client
│   └── bedrock.py          # Bedrock AI client
├── routers/                # API route handlers
│   ├── search.py           # Search endpoints
│   ├── resources.py        # Resource endpoints
│   ├── categories.py       # Category endpoints
│   ├── boilerplate.py      # Code generation
│   ├── users.py            # User management
│   └── health.py           # Health checks
├── services/               # Business logic
│   ├── search.py           # Search service
│   └── ranking.py          # Ranking service
├── models/                 # Data models
│   └── domain.py           # Domain models
├── migrations/             # Database migrations
├── tests/                  # Test suite
├── systemd/                # Systemd service files
├── nginx/                  # Nginx configuration
├── start_server.sh         # Production startup script
└── deploy.sh               # Deployment automation
```

## Redis Caching

### Cache Key Patterns

```
search:{query_hash}              # Search results (5 min TTL)
ranking:{resource_id}:{date}     # Ranking scores (1 hour TTL)
resource:{resource_id}           # Resource metadata (15 min TTL)
user:{user_id}:profile           # User profiles (30 min TTL)
health:{resource_id}             # Health status (10 min TTL)
embedding:{text_hash}            # Embeddings (24 hour TTL)
```

### Cache Operations

```python
# Get cached search results
cached = await redis.get_cached_search(query_hash)

# Cache search results
await redis.cache_search_results(query_hash, results, ttl=300)

# Invalidate cache
await redis.invalidate_resource(resource_id)
await redis.invalidate_pattern("search:*")
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_redis_client.py

# Run property tests
pytest -m property_test
```

## Monitoring

### Health Checks

```bash
# Basic health
curl http://localhost:8000/api/v1/health

# Detailed health
curl http://localhost:8000/api/v1/health/detailed | jq

# Metrics
curl http://localhost:8000/api/v1/metrics | jq

# Cache stats
curl http://localhost:8000/api/v1/cache/stats | jq
```

### Logs

```bash
# Application logs
sudo journalctl -u devstore-api -f

# Nginx logs
sudo tail -f /var/log/nginx/devstore-access.log
sudo tail -f /var/log/nginx/devstore-error.log
```

## Performance

### Expected Performance

- **Cached requests**: < 10ms
- **Database queries**: < 50ms
- **Search with embeddings**: < 500ms
- **Throughput**: 100+ req/s per instance

### Optimization

- Connection pooling for all external services
- Multi-layer caching (Redis + application)
- Async operations for non-blocking I/O
- Proper database indexing
- Gzip compression for responses

## Security

### Best Practices

- All secrets in AWS Secrets Manager
- IAM roles (no hardcoded credentials)
- Input validation with Pydantic
- SQL injection prevention (parameterized queries)
- Rate limiting with Redis
- HTTPS only in production
- Security headers (XSS, CSRF protection)

See [../SECURITY_BEST_PRACTICES.md](../SECURITY_BEST_PRACTICES.md) for details.

## Troubleshooting

### Common Issues

**Cannot connect to Redis:**
```bash
# Test connection
redis-cli -h your-endpoint ping

# Check security group
aws ec2 describe-security-groups --group-ids sg-xxxxx
```

**Database connection errors:**
```bash
# Test PostgreSQL
psql -h your-rds-endpoint -U admin -d devstore

# Check RDS status
aws rds describe-db-clusters --db-cluster-identifier devstore-db-cluster
```

**Service won't start:**
```bash
# Check status
sudo systemctl status devstore-api

# View logs
sudo journalctl -u devstore-api -n 100

# Test manually
python api_gateway.py
```

See [../QUICK_REFERENCE.md](../QUICK_REFERENCE.md) for more troubleshooting commands.

## Documentation

- [EC2 Deployment Guide](../EC2_DEPLOYMENT_GUIDE.md) - Complete deployment walkthrough
- [Security Best Practices](../SECURITY_BEST_PRACTICES.md) - Security guidelines
- [Quick Reference](../QUICK_REFERENCE.md) - Fast reference for common tasks
- [Implementation Summary](../CENTRALIZED_API_SUMMARY.md) - Architecture overview

## Support

For issues and questions:
1. Check the documentation above
2. Review application logs
3. Check AWS service status
4. Consult the troubleshooting guide

## License

MIT
