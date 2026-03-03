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

# Copy environment template
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit backend/.env with your configuration"
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
