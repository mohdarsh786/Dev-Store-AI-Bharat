#!/bin/bash
# Automated deployment script for DevStore API on EC2

set -e

echo "=========================================="
echo "DevStore API Deployment Script"
echo "=========================================="
echo ""

# Configuration
APP_DIR="/home/ubuntu/devstore/backend"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="devstore-api"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as correct user
if [ "$USER" != "ubuntu" ]; then
    log_warn "This script should be run as ubuntu user"
fi

# Step 1: Pull latest code
log_info "Pulling latest code from repository..."
cd $APP_DIR
git pull origin main || log_error "Failed to pull latest code"

# Step 2: Activate virtual environment
log_info "Activating virtual environment..."
source $VENV_DIR/bin/activate || log_error "Failed to activate virtual environment"

# Step 3: Install/update dependencies
log_info "Installing dependencies..."
pip install -r requirements.txt --upgrade || log_error "Failed to install dependencies"

# Step 4: Check environment variables
log_info "Checking environment configuration..."
if [ ! -f "$APP_DIR/.env" ]; then
    log_error ".env file not found! Copy .env.ec2.example to .env and configure it."
    exit 1
fi

# Step 5: Run database migrations
log_info "Running database migrations..."
python run_migrations.py || log_warn "Migration failed or no new migrations"

# Step 6: Test application
log_info "Testing application startup..."
timeout 10 python -c "from api_gateway import app; print('Application imports successfully')" || log_error "Application test failed"

# Step 7: Restart service
log_info "Restarting $SERVICE_NAME service..."
sudo systemctl restart $SERVICE_NAME || log_error "Failed to restart service"

# Step 8: Wait for service to start
log_info "Waiting for service to start..."
sleep 5

# Step 9: Check service status
log_info "Checking service status..."
if sudo systemctl is-active --quiet $SERVICE_NAME; then
    log_info "Service is running"
else
    log_error "Service failed to start"
    sudo systemctl status $SERVICE_NAME
    exit 1
fi

# Step 10: Health check
log_info "Performing health check..."
sleep 2
HEALTH_CHECK=$(curl -s http://localhost:8000/api/v1/health || echo "failed")

if echo "$HEALTH_CHECK" | grep -q "healthy"; then
    log_info "Health check passed"
else
    log_error "Health check failed"
    echo "$HEALTH_CHECK"
    exit 1
fi

# Step 11: Reload Nginx (if needed)
if [ -f "/etc/nginx/sites-enabled/devstore-api.conf" ]; then
    log_info "Reloading Nginx..."
    sudo nginx -t && sudo systemctl reload nginx || log_warn "Nginx reload failed"
fi

echo ""
echo "=========================================="
log_info "Deployment completed successfully!"
echo "=========================================="
echo ""
echo "Service Status:"
sudo systemctl status $SERVICE_NAME --no-pager
echo ""
echo "Recent Logs:"
sudo journalctl -u $SERVICE_NAME -n 20 --no-pager
echo ""
echo "API Endpoints:"
echo "  Health: http://localhost:8000/api/v1/health"
echo "  Docs:   http://localhost:8000/api/docs"
echo ""
