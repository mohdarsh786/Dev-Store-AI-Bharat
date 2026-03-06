#!/bin/bash
# Production startup script for DevStore API on EC2

set -e

echo "Starting DevStore API Server..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Create .env and configure it."
    exit 1
fi

# Start server with Gunicorn
exec gunicorn api_gateway:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --keepalive 5 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
