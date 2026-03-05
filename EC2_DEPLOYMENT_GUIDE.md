# DevStore API - EC2 Deployment Guide

Complete guide for deploying the DevStore centralized API on AWS EC2 with ElastiCache Redis.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [AWS Infrastructure Setup](#aws-infrastructure-setup)
4. [EC2 Instance Setup](#ec2-instance-setup)
5. [Application Deployment](#application-deployment)
6. [Security Configuration](#security-configuration)
7. [Monitoring & Logging](#monitoring--logging)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Internet Gateway                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Application Load Balancer                 │
│                    (Optional - for HA)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      EC2 Instance(s)                         │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Nginx (Reverse Proxy) :80/:443                    │     │
│  │         ↓                                          │     │
│  │  Gunicorn + Uvicorn Workers :8000                  │     │
│  │         ↓                                          │     │
│  │  FastAPI Application (api_gateway.py)             │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                    ↓           ↓           ↓
        ┌───────────┴───────────┴───────────┴───────────┐
        ↓                       ↓                       ↓
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│ ElastiCache  │    │   RDS Aurora     │    │  OpenSearch  │
│   (Redis)    │    │  (PostgreSQL)    │    │   Service    │
└──────────────┘    └──────────────────┘    └──────────────┘
```

---

## Prerequisites

### AWS Account Requirements
- AWS Account with appropriate permissions
- VPC with public and private subnets
- IAM role for EC2 with necessary permissions

### Local Development Tools
- AWS CLI configured
- SSH key pair for EC2 access
- Basic knowledge of Linux administration

---

## AWS Infrastructure Setup

### 1. Create VPC and Subnets

```bash
# Create VPC
aws ec2 create-vpc \
    --cidr-block 10.0.0.0/16 \
    --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=devstore-vpc}]'

# Create public subnet (for EC2)
aws ec2 create-subnet \
    --vpc-id vpc-xxxxx \
    --cidr-block 10.0.1.0/24 \
    --availability-zone us-east-1a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=devstore-public-subnet}]'

# Create private subnets (for RDS, ElastiCache, OpenSearch)
aws ec2 create-subnet \
    --vpc-id vpc-xxxxx \
    --cidr-block 10.0.2.0/24 \
    --availability-zone us-east-1a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=devstore-private-subnet-1}]'

aws ec2 create-subnet \
    --vpc-id vpc-xxxxx \
    --cidr-block 10.0.3.0/24 \
    --availability-zone us-east-1b \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=devstore-private-subnet-2}]'
```

### 2. Create Security Groups

#### EC2 Security Group
```bash
aws ec2 create-security-group \
    --group-name devstore-ec2-sg \
    --description "Security group for DevStore EC2 instances" \
    --vpc-id vpc-xxxxx

# Allow SSH (restrict to your IP)
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxxxx \
    --protocol tcp \
    --port 22 \
    --cidr YOUR_IP/32

# Allow HTTP
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxxxx \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

# Allow HTTPS
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxxxx \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0
```

#### ElastiCache Security Group
```bash
aws ec2 create-security-group \
    --group-name devstore-redis-sg \
    --description "Security group for DevStore ElastiCache" \
    --vpc-id vpc-xxxxx

# Allow Redis from EC2 security group
aws ec2 authorize-security-group-ingress \
    --group-id sg-redis-xxxxx \
    --protocol tcp \
    --port 6379 \
    --source-group sg-ec2-xxxxx
```

#### RDS Security Group
```bash
aws ec2 create-security-group \
    --group-name devstore-rds-sg \
    --description "Security group for DevStore RDS" \
    --vpc-id vpc-xxxxx

# Allow PostgreSQL from EC2 security group
aws ec2 authorize-security-group-ingress \
    --group-id sg-rds-xxxxx \
    --protocol tcp \
    --port 5432 \
    --source-group sg-ec2-xxxxx
```

### 3. Create ElastiCache Redis Cluster

```bash
# Create subnet group
aws elasticache create-cache-subnet-group \
    --cache-subnet-group-name devstore-redis-subnet-group \
    --cache-subnet-group-description "Subnet group for DevStore Redis" \
    --subnet-ids subnet-private-1 subnet-private-2

# Create Redis cluster
aws elasticache create-cache-cluster \
    --cache-cluster-id devstore-redis \
    --engine redis \
    --cache-node-type cache.t3.micro \
    --num-cache-nodes 1 \
    --engine-version 7.0 \
    --cache-subnet-group-name devstore-redis-subnet-group \
    --security-group-ids sg-redis-xxxxx \
    --tags Key=Name,Value=devstore-redis

# Get endpoint after creation
aws elasticache describe-cache-clusters \
    --cache-cluster-id devstore-redis \
    --show-cache-node-info
```

**Note the Primary Endpoint** - you'll need this for the `.env` file.

### 4. Create RDS Aurora PostgreSQL

```bash
# Create DB subnet group
aws rds create-db-subnet-group \
    --db-subnet-group-name devstore-db-subnet-group \
    --db-subnet-group-description "Subnet group for DevStore RDS" \
    --subnet-ids subnet-private-1 subnet-private-2

# Create Aurora cluster
aws rds create-db-cluster \
    --db-cluster-identifier devstore-db-cluster \
    --engine aurora-postgresql \
    --engine-version 15.3 \
    --master-username admin \
    --master-user-password YOUR_SECURE_PASSWORD \
    --db-subnet-group-name devstore-db-subnet-group \
    --vpc-security-group-ids sg-rds-xxxxx \
    --database-name devstore

# Create DB instance
aws rds create-db-instance \
    --db-instance-identifier devstore-db-instance-1 \
    --db-instance-class db.t3.medium \
    --engine aurora-postgresql \
    --db-cluster-identifier devstore-db-cluster
```

### 5. Create OpenSearch Domain

```bash
aws opensearch create-domain \
    --domain-name devstore-search \
    --engine-version OpenSearch_2.11 \
    --cluster-config InstanceType=t3.small.search,InstanceCount=1 \
    --ebs-options EBSEnabled=true,VolumeType=gp3,VolumeSize=20 \
    --vpc-options SubnetIds=subnet-private-1,SecurityGroupIds=sg-opensearch-xxxxx \
    --access-policies '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"AWS": "*"},
            "Action": "es:*",
            "Resource": "arn:aws:es:us-east-1:ACCOUNT_ID:domain/devstore-search/*"
        }]
    }'
```

### 6. Create IAM Role for EC2

```bash
# Create trust policy
cat > ec2-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

# Create role
aws iam create-role \
    --role-name devstore-ec2-role \
    --assume-role-policy-document file://ec2-trust-policy.json

# Attach policies
aws iam attach-role-policy \
    --role-name devstore-ec2-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonRDSFullAccess

aws iam attach-role-policy \
    --role-name devstore-ec2-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonElastiCacheFullAccess

aws iam attach-role-policy \
    --role-name devstore-ec2-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonOpenSearchServiceFullAccess

aws iam attach-role-policy \
    --role-name devstore-ec2-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess

aws iam attach-role-policy \
    --role-name devstore-ec2-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create instance profile
aws iam create-instance-profile \
    --instance-profile-name devstore-ec2-profile

aws iam add-role-to-instance-profile \
    --instance-profile-name devstore-ec2-profile \
    --role-name devstore-ec2-role
```

---

## EC2 Instance Setup

### 1. Launch EC2 Instance

```bash
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.medium \
    --key-name your-key-pair \
    --security-group-ids sg-ec2-xxxxx \
    --subnet-id subnet-public-xxxxx \
    --iam-instance-profile Name=devstore-ec2-profile \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=devstore-api-server}]' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]'
```

### 2. Connect to EC2 Instance

```bash
ssh -i your-key-pair.pem ubuntu@ec2-public-ip
```

### 3. Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Install Nginx
sudo apt install -y nginx

# Install PostgreSQL client
sudo apt install -y postgresql-client

# Install build tools
sudo apt install -y build-essential libssl-dev libffi-dev

# Install Git
sudo apt install -y git
```

---

## Application Deployment

### 1. Clone Repository

```bash
cd /home/ubuntu
git clone https://github.com/your-org/devstore.git
cd devstore/backend
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy example env file
cp .env.ec2.example .env

# Edit with your values
nano .env
```

Fill in the following critical values:
- `DATABASE_URL`: RDS Aurora endpoint
- `REDIS_HOST`: ElastiCache primary endpoint
- `OPENSEARCH_HOST`: OpenSearch domain endpoint
- `S3_BUCKET_*`: Your S3 bucket names

### 5. Run Database Migrations

```bash
python run_migrations.py
```

### 6. Test Application Locally

```bash
# Start server
python api_gateway.py

# In another terminal, test
curl http://localhost:8000/api/v1/health
```

### 7. Configure Systemd Service

```bash
# Copy service file
sudo cp systemd/devstore-api.service /etc/systemd/system/

# Create log directory
sudo mkdir -p /var/log/devstore
sudo chown ubuntu:ubuntu /var/log/devstore

# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable devstore-api

# Start service
sudo systemctl start devstore-api

# Check status
sudo systemctl status devstore-api
```

### 8. Configure Nginx

```bash
# Copy Nginx config
sudo cp nginx/devstore-api.conf /etc/nginx/sites-available/

# Update server_name in config
sudo nano /etc/nginx/sites-available/devstore-api.conf

# Enable site
sudo ln -s /etc/nginx/sites-available/devstore-api.conf /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### 9. Setup SSL with Let's Encrypt (Optional)

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d api.devstore.example.com

# Auto-renewal is configured automatically
```

---

## Security Configuration

### 1. Network Security

- **VPC Isolation**: Keep RDS, ElastiCache, and OpenSearch in private subnets
- **Security Groups**: Use least-privilege access rules
- **NACLs**: Add network ACLs for additional layer of security

### 2. Secrets Management

```bash
# Store secrets in AWS Secrets Manager
aws secretsmanager create-secret \
    --name devstore/database/credentials \
    --secret-string '{"username":"admin","password":"YOUR_PASSWORD"}'

# Update application to fetch from Secrets Manager
```

### 3. IAM Permissions

Ensure EC2 IAM role has minimal required permissions:
- RDS: `rds:DescribeDBInstances`, `rds:Connect`
- ElastiCache: `elasticache:DescribeCacheClusters`
- OpenSearch: `es:ESHttpGet`, `es:ESHttpPost`
- Bedrock: `bedrock:InvokeModel`
- S3: `s3:GetObject`, `s3:PutObject` (specific buckets only)

### 4. Application Security

- Enable rate limiting in Nginx
- Use strong JWT secrets
- Implement request validation
- Enable CORS with specific origins
- Use HTTPS only in production

---

## Monitoring & Logging

### 1. CloudWatch Logs

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb

# Configure agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard
```

### 2. Application Logs

```bash
# View application logs
sudo journalctl -u devstore-api -f

# View Nginx logs
sudo tail -f /var/log/nginx/devstore-access.log
sudo tail -f /var/log/nginx/devstore-error.log
```

### 3. CloudWatch Alarms

```bash
# CPU utilization alarm
aws cloudwatch put-metric-alarm \
    --alarm-name devstore-high-cpu \
    --alarm-description "Alert when CPU exceeds 80%" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=InstanceId,Value=i-xxxxx \
    --evaluation-periods 2

# Memory utilization (requires CloudWatch agent)
# Disk space monitoring
# API error rate monitoring
```

### 4. Performance Monitoring

- Use AWS X-Ray for distributed tracing
- Monitor Redis cache hit rates
- Track API response times
- Monitor database connection pool

---

## Troubleshooting

### Common Issues

#### 1. Cannot Connect to Redis

```bash
# Check security group
aws ec2 describe-security-groups --group-ids sg-redis-xxxxx

# Test connection from EC2
redis-cli -h your-elasticache-endpoint.cache.amazonaws.com ping

# Check application logs
sudo journalctl -u devstore-api | grep -i redis
```

#### 2. Database Connection Errors

```bash
# Test PostgreSQL connection
psql -h your-rds-endpoint.rds.amazonaws.com -U admin -d devstore

# Check RDS security group
# Verify DATABASE_URL in .env
```

#### 3. High Memory Usage

```bash
# Check memory
free -h

# Check Gunicorn workers
ps aux | grep gunicorn

# Reduce workers in start_server.sh if needed
```

#### 4. Slow API Responses

```bash
# Check Redis cache stats
redis-cli -h your-endpoint INFO stats

# Monitor database queries
# Check OpenSearch cluster health
# Review application logs for bottlenecks
```

### Health Checks

```bash
# API health
curl https://api.devstore.example.com/api/v1/health

# Detailed status
curl https://api.devstore.example.com/api/v1/status

# Redis stats
curl https://api.devstore.example.com/api/v1/cache/stats
```

---

## Scaling Considerations

### Horizontal Scaling

1. **Application Load Balancer**: Distribute traffic across multiple EC2 instances
2. **Auto Scaling Group**: Automatically scale based on CPU/memory
3. **Read Replicas**: Add RDS read replicas for read-heavy workloads
4. **Redis Cluster**: Upgrade to Redis cluster mode for higher throughput

### Vertical Scaling

- Upgrade EC2 instance type (t3.medium → t3.large → t3.xlarge)
- Increase RDS instance size
- Scale OpenSearch cluster nodes

---

## Cost Optimization

1. **Use Reserved Instances**: Save up to 72% for predictable workloads
2. **Right-size Resources**: Monitor and adjust instance types
3. **Enable Auto Scaling**: Scale down during low traffic
4. **Use Spot Instances**: For non-critical workloads
5. **Optimize Redis**: Use appropriate eviction policies
6. **S3 Lifecycle Policies**: Move old data to cheaper storage classes

---

## Backup and Disaster Recovery

### Automated Backups

```bash
# RDS automated backups (enabled by default)
aws rds modify-db-cluster \
    --db-cluster-identifier devstore-db-cluster \
    --backup-retention-period 7

# ElastiCache snapshots
aws elasticache create-snapshot \
    --cache-cluster-id devstore-redis \
    --snapshot-name devstore-redis-backup-$(date +%Y%m%d)

# EC2 AMI
aws ec2 create-image \
    --instance-id i-xxxxx \
    --name "devstore-api-backup-$(date +%Y%m%d)"
```

### Disaster Recovery Plan

1. **RTO (Recovery Time Objective)**: 1 hour
2. **RPO (Recovery Point Objective)**: 5 minutes
3. **Multi-AZ Deployment**: Enable for RDS and ElastiCache
4. **Cross-Region Replication**: For critical data in S3

---

## Maintenance

### Regular Tasks

- **Weekly**: Review CloudWatch metrics and logs
- **Monthly**: Update system packages and dependencies
- **Quarterly**: Review and optimize costs
- **Annually**: Disaster recovery drill

### Update Procedure

```bash
# Pull latest code
cd /home/ubuntu/devstore/backend
git pull origin main

# Activate venv
source venv/bin/activate

# Update dependencies
pip install -r requirements.txt --upgrade

# Run migrations
python run_migrations.py

# Restart service
sudo systemctl restart devstore-api

# Verify
curl https://api.devstore.example.com/api/v1/health
```

---

## Support and Resources

- **AWS Documentation**: https://docs.aws.amazon.com/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Redis Documentation**: https://redis.io/documentation
- **Project Repository**: https://github.com/your-org/devstore

---

**Last Updated**: 2026-03-03
**Version**: 2.0.0
