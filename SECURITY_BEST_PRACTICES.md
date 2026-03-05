# DevStore API - Security & Best Practices

Comprehensive security guidelines for prototype-level production deployment on AWS EC2 with ElastiCache.

## Table of Contents
1. [Secrets Management](#secrets-management)
2. [Network Security](#network-security)
3. [IAM Permissions](#iam-permissions)
4. [Application Security](#application-security)
5. [Data Protection](#data-protection)
6. [Logging Strategy](#logging-strategy)
7. [Monitoring Recommendations](#monitoring-recommendations)

---

## Secrets Management

### AWS Secrets Manager Integration

Store sensitive credentials in AWS Secrets Manager instead of environment variables:

```python
# backend/utils/secrets.py
import boto3
import json
from functools import lru_cache

class SecretsManager:
    def __init__(self):
        self.client = boto3.client('secretsmanager', region_name='us-east-1')
    
    @lru_cache(maxsize=10)
    def get_secret(self, secret_name: str) -> dict:
        """Retrieve secret from AWS Secrets Manager with caching"""
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            return json.loads(response['SecretString'])
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {str(e)}")
            raise

# Usage in config.py
secrets_manager = SecretsManager()
db_credentials = secrets_manager.get_secret('devstore/database/credentials')
```

### Secret Rotation

Enable automatic rotation for database credentials:

```bash
# Create rotation Lambda function
aws secretsmanager rotate-secret \
    --secret-id devstore/database/credentials \
    --rotation-lambda-arn arn:aws:lambda:us-east-1:ACCOUNT_ID:function:SecretsManagerRotation \
    --rotation-rules AutomaticallyAfterDays=90
```

### Environment-Specific Secrets

```bash
# Development
devstore/dev/database
devstore/dev/redis
devstore/dev/jwt-secret

# Production
devstore/prod/database
devstore/prod/redis
devstore/prod/jwt-secret
```

### Best Practices

1. **Never commit secrets to Git**: Use `.gitignore` for `.env` files
2. **Use IAM roles**: Prefer IAM roles over access keys for EC2
3. **Rotate regularly**: Set up automatic rotation for all secrets
4. **Least privilege**: Grant minimum required permissions
5. **Audit access**: Enable CloudTrail logging for Secrets Manager

---

## Network Security

### VPC Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          VPC (10.0.0.0/16)                   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Public Subnet (10.0.1.0/24)                       │     │
│  │  - EC2 Instances                                   │     │
│  │  - NAT Gateway                                     │     │
│  │  - Internet Gateway attached                       │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Private Subnet 1 (10.0.2.0/24)                    │     │
│  │  - RDS Aurora                                      │     │
│  │  - ElastiCache Redis                               │     │
│  │  - OpenSearch                                      │     │
│  │  - No direct internet access                       │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Private Subnet 2 (10.0.3.0/24)                    │     │
│  │  - Multi-AZ replicas                               │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Security Group Rules

#### EC2 Security Group (devstore-ec2-sg)

```bash
# Inbound Rules
- Port 22 (SSH): YOUR_IP/32 only
- Port 80 (HTTP): 0.0.0.0/0 (redirect to HTTPS)
- Port 443 (HTTPS): 0.0.0.0/0

# Outbound Rules
- All traffic: 0.0.0.0/0 (for AWS service access)
```

#### ElastiCache Security Group (devstore-redis-sg)

```bash
# Inbound Rules
- Port 6379: Source = devstore-ec2-sg

# Outbound Rules
- None required
```

#### RDS Security Group (devstore-rds-sg)

```bash
# Inbound Rules
- Port 5432: Source = devstore-ec2-sg

# Outbound Rules
- None required
```

#### OpenSearch Security Group (devstore-opensearch-sg)

```bash
# Inbound Rules
- Port 443: Source = devstore-ec2-sg

# Outbound Rules
- None required
```

### Network ACLs

Add an additional layer of security:

```bash
# Public Subnet NACL
Inbound:
100 - Allow HTTP (80) from 0.0.0.0/0
110 - Allow HTTPS (443) from 0.0.0.0/0
120 - Allow SSH (22) from YOUR_IP/32
130 - Allow Ephemeral ports (1024-65535) from 0.0.0.0/0
* - Deny all

Outbound:
100 - Allow all to 0.0.0.0/0
```

### VPC Flow Logs

Enable VPC Flow Logs for network monitoring:

```bash
aws ec2 create-flow-logs \
    --resource-type VPC \
    --resource-ids vpc-xxxxx \
    --traffic-type ALL \
    --log-destination-type cloud-watch-logs \
    --log-group-name /aws/vpc/devstore
```

---

## IAM Permissions

### EC2 Instance Role (Minimal Permissions)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "RDSAccess",
      "Effect": "Allow",
      "Action": [
        "rds:DescribeDBInstances",
        "rds:DescribeDBClusters"
      ],
      "Resource": "arn:aws:rds:us-east-1:ACCOUNT_ID:cluster:devstore-db-cluster"
    },
    {
      "Sid": "ElastiCacheAccess",
      "Effect": "Allow",
      "Action": [
        "elasticache:DescribeCacheClusters"
      ],
      "Resource": "arn:aws:elasticache:us-east-1:ACCOUNT_ID:cluster:devstore-redis"
    },
    {
      "Sid": "OpenSearchAccess",
      "Effect": "Allow",
      "Action": [
        "es:ESHttpGet",
        "es:ESHttpPost",
        "es:ESHttpPut"
      ],
      "Resource": "arn:aws:es:us-east-1:ACCOUNT_ID:domain/devstore-search/*"
    },
    {
      "Sid": "BedrockAccess",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": [
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
        "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1"
      ]
    },
    {
      "Sid": "S3Access",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::devstore-boilerplate-prod/*",
        "arn:aws:s3:::devstore-crawler-data-prod/*"
      ]
    },
    {
      "Sid": "SecretsManagerAccess",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:us-east-1:ACCOUNT_ID:secret:devstore/*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:ACCOUNT_ID:log-group:/aws/devstore/*"
    }
  ]
}
```

### Service Control Policies

Prevent accidental resource deletion:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": [
        "rds:DeleteDBCluster",
        "elasticache:DeleteCacheCluster",
        "es:DeleteDomain"
      ],
      "Resource": "*",
      "Condition": {
        "StringNotEquals": {
          "aws:PrincipalArn": "arn:aws:iam::ACCOUNT_ID:role/AdminRole"
        }
      }
    }
  ]
}
```

---

## Application Security

### Input Validation

```python
# Use Pydantic for strict validation
from pydantic import BaseModel, Field, validator
from typing import Optional, List

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    filters: Optional[dict] = None
    
    @validator('query')
    def sanitize_query(cls, v):
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", ';', '--']
        for char in dangerous_chars:
            v = v.replace(char, '')
        return v.strip()
```

### SQL Injection Prevention

```python
# Always use parameterized queries
async def get_resource(resource_id: str):
    query = "SELECT * FROM resources WHERE id = %s"
    result = await db.execute(query, (resource_id,))
    return result

# NEVER do this:
# query = f"SELECT * FROM resources WHERE id = '{resource_id}'"
```

### XSS Prevention

```python
# Set security headers
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["api.devstore.com"])
app.add_middleware(HTTPSRedirectMiddleware)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response
```

### Rate Limiting

```python
# Implement rate limiting with Redis
from fastapi import HTTPException
import time

class RateLimiter:
    def __init__(self, redis_client, max_requests: int = 100, window: int = 60):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window
    
    async def check_rate_limit(self, user_id: str) -> bool:
        key = f"rate_limit:{user_id}"
        current = await self.redis.get(key)
        
        if current is None:
            await self.redis.set(key, 1, ttl=self.window)
            return True
        
        if int(current) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(self.window)}
            )
        
        await self.redis.client.incr(key)
        return True
```

### Authentication & Authorization

```python
# JWT token validation
from jose import JWTError, jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"  # Store in Secrets Manager
ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=60))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

---

## Data Protection

### Encryption at Rest

1. **RDS Aurora**: Enable encryption when creating cluster
   ```bash
   aws rds create-db-cluster \
       --storage-encrypted \
       --kms-key-id arn:aws:kms:us-east-1:ACCOUNT_ID:key/xxxxx
   ```

2. **ElastiCache**: Enable encryption
   ```bash
   aws elasticache create-cache-cluster \
       --at-rest-encryption-enabled \
       --transit-encryption-enabled
   ```

3. **S3**: Enable default encryption
   ```bash
   aws s3api put-bucket-encryption \
       --bucket devstore-boilerplate-prod \
       --server-side-encryption-configuration '{
           "Rules": [{
               "ApplyServerSideEncryptionByDefault": {
                   "SSEAlgorithm": "AES256"
               }
           }]
       }'
   ```

### Encryption in Transit

1. **HTTPS Only**: Enforce SSL/TLS for all connections
2. **RDS SSL**: Require SSL connections
   ```python
   DATABASE_URL = "postgresql://user:pass@host:5432/db?sslmode=require"
   ```

3. **Redis TLS**: Enable in-transit encryption for ElastiCache

### Data Backup

```bash
# Automated RDS backups
aws rds modify-db-cluster \
    --db-cluster-identifier devstore-db-cluster \
    --backup-retention-period 7 \
    --preferred-backup-window "03:00-04:00"

# ElastiCache snapshots
aws elasticache create-snapshot \
    --cache-cluster-id devstore-redis \
    --snapshot-name daily-backup-$(date +%Y%m%d)
```

### Data Retention Policies

- **Search History**: 90 days
- **User Activity Logs**: 1 year
- **Health Check Data**: 30 days
- **Application Logs**: 30 days
- **Audit Logs**: 7 years (compliance)

---

## Logging Strategy

### Structured Logging

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log(self, level: str, message: str, **kwargs):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "service": "devstore-api",
            "message": message,
            **kwargs
        }
        self.logger.log(
            getattr(logging, level),
            json.dumps(log_entry)
        )

# Usage
logger = StructuredLogger(__name__)
logger.log("INFO", "Search request received", 
           user_id="user-123", 
           query="machine learning API",
           request_id="req-456")
```

### Log Levels

- **DEBUG**: Detailed diagnostic information (development only)
- **INFO**: General informational messages
- **WARNING**: Warning messages for potentially harmful situations
- **ERROR**: Error events that might still allow the application to continue
- **CRITICAL**: Critical events that may cause the application to abort

### What to Log

**DO LOG:**
- All API requests (method, path, status, duration)
- Authentication attempts (success/failure)
- Database queries (with execution time)
- External API calls (Bedrock, OpenSearch)
- Cache hits/misses
- Errors and exceptions with stack traces
- Security events (rate limiting, suspicious activity)

**DON'T LOG:**
- Passwords or API keys
- Credit card numbers
- Personal identifiable information (PII)
- Full request/response bodies (unless debugging)

### CloudWatch Logs Integration

```python
import watchtower
import logging

# Configure CloudWatch handler
logger = logging.getLogger(__name__)
logger.addHandler(watchtower.CloudWatchLogHandler(
    log_group='/aws/devstore/api',
    stream_name='application-logs'
))
```

---

## Monitoring Recommendations

### Key Metrics to Monitor

#### Application Metrics
- **Request Rate**: Requests per second
- **Response Time**: p50, p95, p99 latencies
- **Error Rate**: 4xx and 5xx errors
- **Cache Hit Rate**: Redis cache effectiveness
- **Database Connection Pool**: Active/idle connections

#### Infrastructure Metrics
- **EC2 CPU Utilization**: Target < 70%
- **EC2 Memory Usage**: Target < 80%
- **Disk Usage**: Alert at 80%
- **Network I/O**: Monitor for anomalies

#### Database Metrics
- **RDS CPU**: Target < 70%
- **RDS Connections**: Monitor connection count
- **RDS Read/Write Latency**: Target < 10ms
- **RDS Storage**: Alert at 80% capacity

#### Cache Metrics
- **Redis CPU**: Target < 70%
- **Redis Memory**: Monitor evictions
- **Redis Hit Rate**: Target > 80%
- **Redis Connections**: Monitor connection count

### CloudWatch Dashboards

Create a comprehensive dashboard:

```bash
aws cloudwatch put-dashboard \
    --dashboard-name devstore-api-dashboard \
    --dashboard-body file://dashboard-config.json
```

### Alerting Strategy

#### Critical Alerts (Page immediately)
- API error rate > 10%
- Database connection failures
- Redis unavailable
- EC2 instance down

#### Warning Alerts (Email/Slack)
- API error rate > 5%
- High CPU usage (> 80%)
- High memory usage (> 85%)
- Slow response times (p95 > 1s)

#### Info Alerts (Dashboard only)
- Cache hit rate < 70%
- Unusual traffic patterns
- Scheduled maintenance reminders

### Health Check Endpoints

```python
@app.get("/api/v1/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy"}

@app.get("/api/v1/health/detailed")
async def detailed_health_check(request: Request):
    """Detailed health check with dependency status"""
    checks = {
        "redis": await request.app.state.redis.ping(),
        "database": await request.app.state.db.health_check(),
        "opensearch": await request.app.state.opensearch.health_check()
    }
    
    all_healthy = all(checks.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": time.time()
    }
```

### Performance Monitoring

Use AWS X-Ray for distributed tracing:

```python
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.ext.fastapi.middleware import XRayMiddleware

app.add_middleware(XRayMiddleware, recorder=xray_recorder)

@xray_recorder.capture('search_operation')
async def search(query: str):
    # Your search logic
    pass
```

---

## Incident Response Plan

### 1. Detection
- CloudWatch alarms trigger
- User reports issues
- Monitoring dashboard shows anomalies

### 2. Assessment
- Check health endpoints
- Review CloudWatch logs
- Verify infrastructure status

### 3. Response
- Scale resources if needed
- Restart services
- Rollback recent changes
- Enable maintenance mode

### 4. Recovery
- Restore from backups if needed
- Verify all systems operational
- Monitor for recurring issues

### 5. Post-Mortem
- Document incident timeline
- Identify root cause
- Implement preventive measures
- Update runbooks

---

## Compliance Considerations

### GDPR Compliance
- Implement data deletion endpoints
- Provide data export functionality
- Maintain audit logs
- Encrypt personal data

### Data Residency
- Store data in appropriate AWS regions
- Use S3 bucket policies for geo-restrictions
- Document data flow

### Audit Trail
- Log all data access
- Track configuration changes
- Maintain immutable logs

---

## Security Checklist

### Pre-Deployment
- [ ] All secrets stored in Secrets Manager
- [ ] IAM roles follow least privilege
- [ ] Security groups properly configured
- [ ] SSL/TLS certificates installed
- [ ] Database encryption enabled
- [ ] Backup strategy implemented
- [ ] Monitoring and alerting configured

### Post-Deployment
- [ ] Penetration testing completed
- [ ] Security audit performed
- [ ] Incident response plan documented
- [ ] Team trained on security procedures
- [ ] Regular security reviews scheduled

### Ongoing
- [ ] Weekly security log reviews
- [ ] Monthly dependency updates
- [ ] Quarterly security assessments
- [ ] Annual penetration testing

---

**Last Updated**: 2026-03-03
**Version**: 1.0.0
