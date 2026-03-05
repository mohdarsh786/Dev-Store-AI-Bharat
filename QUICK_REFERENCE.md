# DevStore API - Quick Reference Guide

Fast reference for common tasks and commands.

## Table of Contents
- [API Endpoints](#api-endpoints)
- [Environment Variables](#environment-variables)
- [Common Commands](#common-commands)
- [Redis Cache Keys](#redis-cache-keys)
- [Troubleshooting](#troubleshooting)

---

## API Endpoints

### Base URL
```
Development: http://localhost:8000
Production: https://api.devstore.example.com
```

### Health & Status
```bash
# Basic health check
GET /api/v1/health

# Detailed health with dependencies
GET /api/v1/health/detailed

# Application metrics
GET /api/v1/metrics

# Cache statistics
GET /api/v1/cache/stats
```

### Search
```bash
# Semantic search
POST /api/v1/search
{
  "query": "machine learning API for image classification",
  "filters": {
    "pricing_types": ["free"],
    "resource_types": ["api", "model"]
  }
}
```

### Resources
```bash
# List resources
GET /api/v1/resources?page=1&page_size=20&type=api&pricing=free

# Get resource details
GET /api/v1/resources/{resource_id}

# Get resource health status
GET /api/v1/resources/{resource_id}/health
```

### Categories
```bash
# List all categories
GET /api/v1/categories

# Get resources in category
GET /api/v1/categories/{category_id}/resources?subcategory=top-free
```

### Boilerplate
```bash
# Generate boilerplate code
POST /api/v1/boilerplate/generate
{
  "resource_ids": ["uuid1", "uuid2"],
  "language": "python",
  "include_tests": true,
  "include_docker": false
}
```

### Users
```bash
# Get user profile
GET /api/v1/users/profile

# Update user profile
PUT /api/v1/users/profile
{
  "preferred_language": "en",
  "tech_stack": ["python", "react", "aws"]
}

# Track user action
POST /api/v1/users/track
{
  "action": "view",
  "resource_id": "uuid",
  "metadata": {}
}

# Get user history
GET /api/v1/users/history?limit=50

# Get recommendations
GET /api/v1/users/recommendations?limit=10
```

---

## Environment Variables

### Required Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/devstore

# Redis
REDIS_HOST=your-elasticache-endpoint.cache.amazonaws.com
REDIS_PORT=6379

# OpenSearch
OPENSEARCH_HOST=your-opensearch-endpoint.us-east-1.es.amazonaws.com

# AWS Region
AWS_REGION=us-east-1
```

### Optional Variables
```bash
# Redis
REDIS_PASSWORD=
REDIS_DB=0
REDIS_POOL_SIZE=50

# API
API_RATE_LIMIT=100
API_TIMEOUT=30
CORS_ORIGINS=["*"]

# Server
ENVIRONMENT=production
LOG_LEVEL=INFO
SERVER_PORT=8000
WORKERS=4
```

---

## Common Commands

### Local Development
```bash
# Start development server
cd backend
source venv/bin/activate
python api_gateway.py

# Run with auto-reload
uvicorn api_gateway:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment
```bash
# Start with Gunicorn
./start_server.sh

# Or manually
gunicorn api_gateway:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

### Systemd Service
```bash
# Start service
sudo systemctl start devstore-api

# Stop service
sudo systemctl stop devstore-api

# Restart service
sudo systemctl restart devstore-api

# Check status
sudo systemctl status devstore-api

# View logs
sudo journalctl -u devstore-api -f
```

### Database Migrations
```bash
# Run migrations
python run_migrations.py

# Rollback migrations
python run_migrations.py --rollback
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_search.py

# Run property tests only
pytest -m property_test
```

---

## Redis Cache Keys

### Key Patterns
```
search:{query_hash}              # Search results (5 min TTL)
ranking:{resource_id}:{date}     # Ranking scores (1 hour TTL)
resource:{resource_id}           # Resource metadata (15 min TTL)
user:{user_id}:profile           # User profiles (30 min TTL)
health:{resource_id}             # Health status (10 min TTL)
embedding:{text_hash}            # Embeddings (24 hour TTL)
```

### Cache Operations
```bash
# Connect to Redis
redis-cli -h your-elasticache-endpoint.cache.amazonaws.com

# Get cache stats
INFO stats

# Check specific key
GET search:abc123

# List all keys (use carefully in production!)
KEYS *

# Delete specific key
DEL search:abc123

# Delete pattern
EVAL "return redis.call('del', unpack(redis.call('keys', ARGV[1])))" 0 "search:*"

# Flush all (DANGEROUS!)
FLUSHDB
```

### Cache Invalidation
```bash
# Via API
POST /api/v1/cache/flush

# Via Python
from clients.redis_client import RedisClient
redis = RedisClient()
await redis.invalidate_pattern("search:*")
await redis.invalidate_resource("resource-id")
```

---

## Troubleshooting

### Cannot Connect to Redis
```bash
# Check Redis is running
aws elasticache describe-cache-clusters \
    --cache-cluster-id devstore-redis \
    --show-cache-node-info

# Test connection from EC2
redis-cli -h your-endpoint ping

# Check security group
aws ec2 describe-security-groups --group-ids sg-xxxxx

# Check application logs
sudo journalctl -u devstore-api | grep -i redis
```

### Database Connection Errors
```bash
# Test PostgreSQL connection
psql -h your-rds-endpoint -U admin -d devstore

# Check RDS status
aws rds describe-db-clusters --db-cluster-identifier devstore-db-cluster

# Check connection pool
# In Python shell:
from clients.database import DatabaseClient
db = DatabaseClient()
await db.health_check()
```

### High Memory Usage
```bash
# Check memory
free -h

# Check process memory
ps aux --sort=-%mem | head

# Check Redis memory
redis-cli -h your-endpoint INFO memory

# Restart service
sudo systemctl restart devstore-api
```

### Slow API Responses
```bash
# Check API metrics
curl https://api.devstore.example.com/api/v1/metrics

# Check cache hit rate
curl https://api.devstore.example.com/api/v1/cache/stats

# Monitor logs
sudo journalctl -u devstore-api -f | grep -E "Time:|Duration:"

# Check database slow queries
# Connect to RDS and run:
SELECT * FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;
```

### Service Won't Start
```bash
# Check service status
sudo systemctl status devstore-api

# Check logs
sudo journalctl -u devstore-api -n 100

# Check if port is in use
sudo lsof -i :8000

# Check environment variables
sudo systemctl show devstore-api | grep Environment

# Test manually
cd /home/ubuntu/devstore/backend
source venv/bin/activate
python api_gateway.py
```

### SSL Certificate Issues
```bash
# Check certificate
sudo certbot certificates

# Renew certificate
sudo certbot renew

# Test Nginx config
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

---

## Performance Tuning

### Gunicorn Workers
```bash
# Formula: (2 x CPU cores) + 1
# For 2 CPU cores: 5 workers
# For 4 CPU cores: 9 workers

# Update in start_server.sh
--workers 9
```

### Redis Connection Pool
```bash
# In .env
REDIS_POOL_SIZE=50  # Increase for high traffic

# Monitor connections
redis-cli -h your-endpoint CLIENT LIST | wc -l
```

### Database Connection Pool
```bash
# In .env
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# Monitor connections
SELECT count(*) FROM pg_stat_activity;
```

---

## Monitoring Commands

### Check System Resources
```bash
# CPU usage
top -bn1 | grep "Cpu(s)"

# Memory usage
free -h

# Disk usage
df -h

# Network connections
netstat -an | grep :8000 | wc -l
```

### Check Application Health
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Detailed health
curl http://localhost:8000/api/v1/health/detailed | jq

# Metrics
curl http://localhost:8000/api/v1/metrics | jq
```

### Check Logs
```bash
# Application logs
sudo journalctl -u devstore-api -f

# Nginx access logs
sudo tail -f /var/log/nginx/devstore-access.log

# Nginx error logs
sudo tail -f /var/log/nginx/devstore-error.log

# System logs
sudo tail -f /var/log/syslog
```

---

## AWS CLI Commands

### ElastiCache
```bash
# Describe cluster
aws elasticache describe-cache-clusters \
    --cache-cluster-id devstore-redis \
    --show-cache-node-info

# Create snapshot
aws elasticache create-snapshot \
    --cache-cluster-id devstore-redis \
    --snapshot-name backup-$(date +%Y%m%d)
```

### RDS
```bash
# Describe cluster
aws rds describe-db-clusters \
    --db-cluster-identifier devstore-db-cluster

# Create snapshot
aws rds create-db-cluster-snapshot \
    --db-cluster-identifier devstore-db-cluster \
    --db-cluster-snapshot-identifier backup-$(date +%Y%m%d)
```

### EC2
```bash
# Describe instance
aws ec2 describe-instances \
    --instance-ids i-xxxxx

# Create AMI
aws ec2 create-image \
    --instance-id i-xxxxx \
    --name "devstore-api-$(date +%Y%m%d)"
```

### CloudWatch
```bash
# Get CPU metrics
aws cloudwatch get-metric-statistics \
    --namespace AWS/EC2 \
    --metric-name CPUUtilization \
    --dimensions Name=InstanceId,Value=i-xxxxx \
    --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
    --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
    --period 300 \
    --statistics Average
```

---

## Security Commands

### Check Open Ports
```bash
# List listening ports
sudo netstat -tulpn | grep LISTEN

# Check firewall rules
sudo ufw status
```

### SSL/TLS
```bash
# Test SSL certificate
openssl s_client -connect api.devstore.example.com:443

# Check certificate expiry
echo | openssl s_client -connect api.devstore.example.com:443 2>/dev/null | openssl x509 -noout -dates
```

### Secrets
```bash
# Get secret from Secrets Manager
aws secretsmanager get-secret-value \
    --secret-id devstore/prod/database \
    --query SecretString \
    --output text
```

---

## Backup & Restore

### Create Backup
```bash
# Database backup
pg_dump -h your-rds-endpoint -U admin devstore > backup.sql

# Redis backup (automatic snapshots)
aws elasticache create-snapshot \
    --cache-cluster-id devstore-redis \
    --snapshot-name manual-backup

# Application files
tar -czf devstore-backup-$(date +%Y%m%d).tar.gz /home/ubuntu/devstore
```

### Restore from Backup
```bash
# Database restore
psql -h your-rds-endpoint -U admin devstore < backup.sql

# Redis restore
aws elasticache create-cache-cluster \
    --cache-cluster-id devstore-redis-restored \
    --snapshot-name manual-backup
```

---

## Useful Aliases

Add to `~/.bashrc`:
```bash
# DevStore aliases
alias devstore-start='sudo systemctl start devstore-api'
alias devstore-stop='sudo systemctl stop devstore-api'
alias devstore-restart='sudo systemctl restart devstore-api'
alias devstore-status='sudo systemctl status devstore-api'
alias devstore-logs='sudo journalctl -u devstore-api -f'
alias devstore-health='curl http://localhost:8000/api/v1/health | jq'
alias devstore-metrics='curl http://localhost:8000/api/v1/metrics | jq'
```

---

**Last Updated**: 2026-03-03
**Version**: 1.0.0
