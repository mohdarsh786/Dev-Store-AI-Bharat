#!/bin/bash

# Dev-Store Backend Setup Script

echo "================================================"
echo " Dev-Store Backend Setup (Linux/Mac)"
echo "================================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Please install Python 3.11+"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "[OK] Python $python_version detected"
echo ""

# Create virtual environment
if [ -d "venv" ]; then
    echo "[INFO] Virtual environment already exists"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
    echo "[OK] Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo ""

# Install dependencies
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi
echo "[OK] Dependencies installed"
echo ""

# Copy environment template
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "Creating .env file..."
        cp .env.example .env
        echo "[OK] .env created. Please edit .env with your AWS credentials"
    else
        echo "[WARNING] .env.example not found"
    fi
else
    echo "[INFO] .env already exists"
fi
echo ""

# Clean cache
echo "Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "[OK] Cache cleaned"
echo ""

echo "================================================"
echo " Backend Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your AWS credentials"
echo "  2. Test connections: cd tests && python test_connections_simple.py"
echo "  3. Create OpenSearch index: python setup_opensearch_index.py"
echo "  4. Start server: uvicorn main:app --reload --port 8000"
echo ""
echo "API docs will be at: http://localhost:8000/docs"
echo ""
