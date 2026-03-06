# Complete Deployment Guide - DevStore

## 🎯 Choose Your Deployment Path

### Path 1: Quick Local Demo (5 minutes) ✅ READY NOW
- No AWS setup needed
- Works with mock data
- Perfect for testing and demos

### Path 2: Local with AWS Services (2-3 hours)
- Connect to real AWS services
- Full functionality
- Run on your local machine

### Path 3: Production EC2 Deployment (4-6 hours)
- Full production setup
- Scalable and reliable
- Public internet access

---

## 📋 Path 1: Quick Local Demo (START HERE)

### Prerequisites
- ✅ Python 3.11+ installed
- ✅ Node.js 18+ installed
- ✅ Git installed

### Step 1: Start Backend (2 minutes)

```bash
# Navigate to backend
cd Dev-Store-AI-Bharat/backend

# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source venv/bin/activate  # Linux/Mac

# Start server
uvicorn api_gateway:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
📦 AWS not configured - using mock data
```

**Verify:** Open http://localhost:8000/api/v1/health

### Step 2: Start Frontend (2 minutes)

```bash
# Open new terminal
cd Dev-Store-AI-Bharat/frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

**Expected Output:**
```
VITE v5.0.8  ready in 500 ms
➜  Local:   http://localhost:5173/
```

**Verify:** Open http://localhost:5173

### Step 3: Test the Application (1 minute)

1. Open http://localhost:5173
2. Try searching: "image generation API"
3. Browse mock resources
4. Check that everything loads

**✅ Success!** You now have a working local demo.

---

## 📋 Path 2: Local with AWS Services

### Prerequisites
- ✅ Path 1 completed
- ✅ AWS Account with billing enabled
- ✅ AWS CLI installed and configured

### Step 1: Create AWS Infrastructure (1-2 hours)

#### 1.1 Configure AWS CLI
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Default region: us-east-1
# Default output format: json
```

#### 1.2 Create RDS PostgreSQL Database
```bash
# Create database instance
aws rds create-db-instance \
  --db-instance-identifier devstore-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --engine-version 15.4 \
  --master-username devstore_admin \
  --master-user-password "YourSecurePassword123!" \
  --allocated-storage 20 \
  --vpc-security-group-ids sg-xxxxxxxx \
  --publicly-accessible \
  --backup-retention-period 7 \
  --tags Key=Project,Value=DevStore

# Wait for creation (5-10 minutes)
aws rds wait db-instance-available \
  --db-instance-identifier devstore-db

# Get endpoint
aws rds describe-db-instances \
  --db-instance-identifier devstore-db \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text
```

**Save the endpoint:** `devstore-db.xxxxxxxxx.us-east-1.rds.amazonaws.com`

#### 1.3 Create ElastiCache Redis Cluster
```bash
# Create Redis cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id devstore-cache \
  --cache-node-type cache.t3.micro \
  --engine redis \
  --num-cache-nodes 1 \
  --security-group-ids sg-xxxxxxxx \
  --tags Key=Project,Value=DevStore

# Wait for creation (5-10 minutes)
aws elasticache wait cache-cluster-available \
  --cache-cluster-id devstore-cache

# Get endpoint
aws elasticache describe-cache-clusters \
  --cache-cluster-id devstore-cache \
  --show-cache-node-info \
  --query 'CacheClusters[0].CacheNodes[0].Endpoint.Address' \
  --output text
```

**Save the endpoint:** `devstore-cache.xxxxxx.0001.use1.cache.amazonaws.com`

#### 1.4 Create OpenSearch Domain
```bash
# Create OpenSearch domain
aws opensearch create-domain \
  --domain-name devstore-search \
  --engine-version OpenSearch_2.11 \
  --cluster-config InstanceType=t3.small.search,InstanceCount=1 \
  --ebs-options EBSEnabled=true,VolumeType=gp3,VolumeSize=10 \
  --access-policies '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"AWS": "*"},
      "Action": "es:*",
      "Resource": "arn:aws:es:us-east-1:*:domain/devstore-search/*"
    }]
  }' \
  --tags Key=Project,Value=DevStore

# Wait for creation (10-15 minutes)
aws opensearch wait domain-available \
  --domain-name devstore-search

# Get endpoint
aws opensearch describe-domain \
  --domain-name devstore-search \
  --query 'DomainStatus.Endpoint' \
  --output text
```

**Save the endpoint:** `search-devstore-search-xxxxxxxxx.us-east-1.es.amazonaws.com`

#### 1.5 Create S3 Buckets
```bash
# Create buckets
aws s3 mb s3://devstore-boilerplate-$(date +%s)
aws s3 mb s3://devstore-crawler-$(date +%s)

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket devstore-boilerplate-xxxxxx \
  --versioning-configuration Status=Enabled
```

#### 1.6 Enable Bedrock Model Access
```bash
# List available models
aws bedrock list-foundation-models --region us-east-1

# Request access via AWS Console:
# 1. Go to https://console.aws.amazon.com/bedrock
# 2. Click "Model access" in left menu
# 3. Click "Manage model access"
# 4. Enable: Claude 3 Sonnet, Titan Embeddings
# 5. Click "Save changes"
```

### Step 2: Configure Backend (15 minutes)

#### 2.1 Update .env File
```bash
cd Dev-Store-AI-Bharat/backend
nano .env  # or use your preferred editor
```

**Update with your AWS endpoints:**
```env
# Database Configuration
DATABASE_URL=postgresql://devstore_admin:YourSecurePassword123!@devstore-db.xxxxxxxxx.us-east-1.rds.amazonaws.com:5432/postgres
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# ElastiCache Configuration
REDIS_HOST=devstore-cache.xxxxxx.0001.use1.cache.amazonaws.com
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_POOL_SIZE=50
REDIS_SOCKET_TIMEOUT=5

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key

# OpenSearch Configuration
OPENSEARCH_HOST=search-devstore-search-xxxxxxxxx.us-east-1.es.amazonaws.com
OPENSEARCH_PORT=443
OPENSEARCH_USE_SSL=true
OPENSEARCH_INDEX_NAME=devstore_resources

# Bedrock Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1

# S3 Configuration
S3_BUCKET_BOILERPLATE=devstore-boilerplate-xxxxxx
S3_BUCKET_CRAWLER_DATA=devstore-crawler-xxxxxx

# API Configuration
API_RATE_LIMIT=100
API_TIMEOUT=30

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# CORS Origins
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

#### 2.2 Run Database Migrations
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows
# OR
source venv/bin/activate  # Linux/Mac

# Run migrations
python run_migrations.py
```

**Expected Output:**
```
Running migration: 001_create_resources_table.sql
Running migration: 002_create_categories_tables.sql
...
✅ All migrations completed successfully
```

#### 2.3 Test AWS Connectivity
```bash
# Test database connection
python -c "from clients.database import DatabaseClient; import asyncio; asyncio.run(DatabaseClient().connect()); print('✅ Database connected')"

# Test Redis connection
python -c "from clients.redis_client import RedisClient; import asyncio; asyncio.run(RedisClient().connect()); print('✅ Redis connected')"

# Test OpenSearch connection
python -c "from clients.opensearch import OpenSearchClient; import asyncio; asyncio.run(OpenSearchClient().connect()); print('✅ OpenSearch connected')"
```

### Step 3: Start with AWS Services (5 minutes)

```bash
# Start backend
uvicorn api_gateway:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
✅ AWS services configured - using real search
INFO:     Connected to Redis at devstore-cache...
INFO:     Async database pool initialized
INFO:     OpenSearch client initialized successfully
```

**Verify:** 
```bash
curl http://localhost:8000/api/v1/status
```

**Expected Response:**
```json
{
  "status": "healthy",
  "dependencies": {
    "redis": "healthy",
    "database": "healthy",
    "opensearch": "healthy",
    "bedrock": "healthy"
  }
}
```

**✅ Success!** You now have full AWS integration running locally.

---

## 📋 Path 3: Production EC2 Deployment

### Prerequisites
- ✅ Path 2 completed (AWS services created)
- ✅ EC2 instance launched
- ✅ Domain name (optional)
- ✅ SSL certificate (optional)

### Step 1: Launch EC2 Instance (30 minutes)

#### 1.1 Create EC2 Instance
```bash
# Create key pair
aws ec2 create-key-pair \
  --key-name devstore-key \
  --query 'KeyMaterial' \
  --output text > devstore-key.pem

chmod 400 devstore-key.pem

# Launch instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --key-name devstore-key \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=DevStore-API}]' \
  --user-data file://user-data.sh
```

#### 1.2 Configure Security Group
```bash
# Allow HTTP
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxx \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0

# Allow HTTPS
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxx \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0

# Allow SSH (your IP only)
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxx \
  --protocol tcp \
  --port 22 \
  --cidr YOUR_IP/32
```

### Step 2: Setup EC2 Instance (1 hour)

#### 2.1 Connect to EC2
```bash
# Get instance IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=DevStore-API" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

# SSH to instance
ssh -i devstore-key.pem ubuntu@$INSTANCE_IP
```

#### 2.2 Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3-pip

# Install Nginx
sudo apt install -y nginx

# Install Git
sudo apt install -y git

# Install PostgreSQL client (for migrations)
sudo apt install -y postgresql-client
```

#### 2.3 Clone Repository
```bash
# Create app directory
sudo mkdir -p /home/ubuntu/devstore
sudo chown ubuntu:ubuntu /home/ubuntu/devstore

# Clone repository
cd /home/ubuntu/devstore
git clone https://github.com/your-username/Dev-Store-AI-Bharat.git .
```

#### 2.4 Setup Python Environment
```bash
cd /home/ubuntu/devstore/backend

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2.5 Configure Environment
```bash
# Copy .env.example to .env
cp .env.example .env

# Edit .env with your AWS endpoints
nano .env
```

**Use the same configuration from Path 2, Step 2.1**

#### 2.6 Run Migrations
```bash
python run_migrations.py
```

### Step 3: Setup Systemd Service (15 minutes)

#### 3.1 Create Log Directory
```bash
sudo mkdir -p /var/log/devstore
sudo chown ubuntu:ubuntu /var/log/devstore
```

#### 3.2 Install Systemd Service
```bash
# Copy service file
sudo cp systemd/devstore-api.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable devstore-api

# Start service
sudo systemctl start devstore-api

# Check status
sudo systemctl status devstore-api
```

**Expected Output:**
```
● devstore-api.service - DevStore API Service
   Loaded: loaded (/etc/systemd/system/devstore-api.service)
   Active: active (running)
```

### Step 4: Setup Nginx (15 minutes)

#### 4.1 Configure Nginx
```bash
# Copy nginx config
sudo cp nginx/devstore-api.conf /etc/nginx/sites-available/

# Update server_name in config
sudo nano /etc/nginx/sites-available/devstore-api.conf
# Change: server_name api.devstore.example.com;
# To: server_name your-domain.com;

# Enable site
sudo ln -s /etc/nginx/sites-available/devstore-api.conf /etc/nginx/sites-enabled/

# Remove default site
sudo rm /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

#### 4.2 Setup SSL (Optional - Let's Encrypt)
```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is configured automatically
```

### Step 5: Deploy Frontend (30 minutes)

#### 5.1 Build Frontend
```bash
# On your local machine
cd Dev-Store-AI-Bharat/frontend

# Update API URL
nano .env
# Change: VITE_API_URL=http://localhost:8000
# To: VITE_API_URL=https://your-domain.com

# Build
npm run build
```

#### 5.2 Deploy to S3 + CloudFront (Recommended)
```bash
# Create S3 bucket for frontend
aws s3 mb s3://devstore-frontend-$(date +%s)

# Enable static website hosting
aws s3 website s3://devstore-frontend-xxxxxx \
  --index-document index.html \
  --error-document index.html

# Upload build
aws s3 sync dist/ s3://devstore-frontend-xxxxxx --delete

# Create CloudFront distribution (optional)
aws cloudfront create-distribution \
  --origin-domain-name devstore-frontend-xxxxxx.s3.amazonaws.com \
  --default-root-object index.html
```

### Step 6: Verify Deployment (10 minutes)

#### 6.1 Test Backend
```bash
# Health check
curl https://your-domain.com/api/v1/health

# Search test
curl -X POST https://your-domain.com/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "image generation", "limit": 5}'
```

#### 6.2 Test Frontend
Open your frontend URL in browser and verify:
- ✅ Page loads
- ✅ Search works
- ✅ Resources display
- ✅ No console errors

#### 6.3 Monitor Logs
```bash
# Backend logs
sudo journalctl -u devstore-api -f

# Nginx access logs
sudo tail -f /var/log/nginx/devstore-access.log

# Nginx error logs
sudo tail -f /var/log/nginx/devstore-error.log

# Application logs
sudo tail -f /var/log/devstore/error.log
```

**✅ Success!** Your application is now deployed to production!

---

## 🔄 Continuous Deployment

### Automated Deployment Script

The `deploy.sh` script automates updates:

```bash
# SSH to EC2
ssh -i devstore-key.pem ubuntu@your-ec2-ip

# Run deployment
cd /home/ubuntu/devstore/backend
./deploy.sh
```

**What it does:**
1. Pulls latest code
2. Installs dependencies
3. Runs migrations
4. Restarts service
5. Performs health check
6. Reloads Nginx

### Setup GitHub Actions (Optional)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to EC2

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to EC2
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.EC2_KEY }}
          script: |
            cd /home/ubuntu/devstore/backend
            ./deploy.sh
```

---

## 📊 Monitoring & Maintenance

### Health Checks
```bash
# Backend health
curl https://your-domain.com/api/v1/health

# Service status
sudo systemctl status devstore-api

# Check logs
sudo journalctl -u devstore-api --since "1 hour ago"
```

### Performance Monitoring
```bash
# CPU and memory usage
htop

# Disk usage
df -h

# Network connections
netstat -tulpn | grep 8000
```

### Database Maintenance
```bash
# Backup database
pg_dump -h your-rds-endpoint -U devstore_admin -d postgres > backup.sql

# Restore database
psql -h your-rds-endpoint -U devstore_admin -d postgres < backup.sql
```

---

## 🆘 Troubleshooting

### Issue: Service won't start
```bash
# Check logs
sudo journalctl -u devstore-api -n 50

# Check if port is in use
sudo lsof -i :8000

# Restart service
sudo systemctl restart devstore-api
```

### Issue: Database connection failed
```bash
# Test connection
psql -h your-rds-endpoint -U devstore_admin -d postgres

# Check security group allows EC2 IP
aws ec2 describe-security-groups --group-ids sg-xxxxxxxx
```

### Issue: OpenSearch connection failed
```bash
# Test connection
curl https://your-opensearch-endpoint/_cluster/health

# Check access policy
aws opensearch describe-domain --domain-name devstore-search
```

### Issue: High memory usage
```bash
# Reduce Gunicorn workers
sudo nano /etc/systemd/system/devstore-api.service
# Change: --workers 4
# To: --workers 2

sudo systemctl daemon-reload
sudo systemctl restart devstore-api
```

---

## 📈 Scaling Considerations

### Horizontal Scaling
- Add Application Load Balancer
- Launch multiple EC2 instances
- Use Auto Scaling Groups

### Database Scaling
- Upgrade RDS instance class
- Enable read replicas
- Use connection pooling (already configured)

### Caching
- Redis already configured
- Add CloudFront for static assets
- Enable API response caching

---

## 💰 Cost Optimization

### Development
- Use t3.micro instances
- Stop instances when not in use
- Use AWS Free Tier

### Production
- Use Reserved Instances (save 30-70%)
- Enable auto-scaling
- Use S3 lifecycle policies
- Monitor with AWS Cost Explorer

---

## ✅ Deployment Checklist

### Before Deployment
- [ ] All code issues fixed
- [ ] Dependencies installed
- [ ] Environment variables configured
- [ ] Database migrations tested
- [ ] AWS services created
- [ ] Security groups configured
- [ ] SSL certificate obtained (production)

### After Deployment
- [ ] Health checks passing
- [ ] Search functionality working
- [ ] Database queries working
- [ ] Redis caching working
- [ ] Logs being written
- [ ] Monitoring configured
- [ ] Backup strategy in place
- [ ] Documentation updated

---

## 🎉 You're Done!

Your DevStore application is now fully deployed and running in production!

**Next Steps:**
- Monitor application performance
- Set up alerts for errors
- Configure automated backups
- Plan for scaling
- Gather user feedback
