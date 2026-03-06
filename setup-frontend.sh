#!/bin/bash

# DevStore Frontend Setup Script

echo "Setting up DevStore Frontend..."

# Check Node version
node_version=$(node --version 2>&1)
echo "Node version: $node_version"

# Navigate to frontend directory
cd frontend

# Install dependencies
echo "Installing dependencies..."
npm install

# Create single environment file
if [ ! -f .env ]; then
    echo "Creating frontend/.env..."
    cat > .env << 'EOF'
# DevStore Frontend Environment
VITE_API_URL=http://localhost:8000
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_API_TIMEOUT=30000
VITE_ENABLE_AUTH=false
VITE_ENABLE_ANALYTICS=false
VITE_ENVIRONMENT=development
EOF
    echo "Please edit frontend/.env with your deployed API endpoint"
fi

echo ""
echo "Frontend setup complete!"
echo ""
echo "To start the development server, run:"
echo "  cd frontend && npm run dev"
echo ""
echo "To build for production, run:"
echo "  cd frontend && npm run build"
echo ""
