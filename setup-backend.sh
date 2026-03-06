#!/bin/bash

# DevStore Backend Setup Script

echo "Setting up DevStore Backend..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Navigate to backend directory
cd backend

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create single environment file
if [ ! -f .env ]; then
    echo "Creating backend/.env..."
    cat > .env << 'EOF'
# DevStore Backend Environment
DATABASE_URL=postgresql://user:password@localhost:5432/devstore
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=443
AWS_REGION=us-east-1
S3_BUCKET_BOILERPLATE=devstore-boilerplate
S3_BUCKET_CRAWLER_DATA=devstore-crawler-data
ENVIRONMENT=development
LOG_LEVEL=INFO
EOF
    echo "Please edit backend/.env with your real configuration"
fi

echo ""
echo "Backend setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  cd backend && source venv/bin/activate"
echo ""
echo "To start the development server, run:"
echo "  uvicorn main:app --reload --port 8000"
echo ""
