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

# Copy environment template
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit frontend/.env with your configuration"
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
