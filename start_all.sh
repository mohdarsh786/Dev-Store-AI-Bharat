#!/bin/bash

# DevStore Main Start Script
# This will start BOTH the FastAPI backend and the Vite frontend simultaneously.

echo "========================================="
echo "🚀 Starting DevStore Services"
echo "========================================="

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down DEVStore services..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT SIGTERM

echo ""
echo "📦 Setting up Backend (FastAPI)..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "   Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies (quietly)
echo "   Installing Python dependencies..."
pip install -r requirements.txt -q

# Set up environment variables if they don't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "   Copying .env.example to .env..."
    cp .env.example .env
fi

# Start backend server
echo "🟢 Starting FastAPI server on port 8000..."
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

echo ""
echo "📦 Setting up Frontend (React/Vite)..."
cd frontend

# Install Node dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "   Installing Node dependencies..."
    npm install --silent
fi

# Set up frontend environment variables if they don't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "   Copying .env.example to .env..."
    cp .env.example .env
fi

# Start frontend server
echo "🟢 Starting React server on port 3000..."
npm run dev -- --port 3000 &
FRONTEND_PID=$!
cd ..

echo ""
echo "========================================="
echo "✨ All services started!"
echo "========================================="
echo "📱 Frontend Dashboard : http://localhost:3000"
echo "🔧 Backend API Docs   : http://localhost:8000/docs"
echo "👉 Press Ctrl+C to stop both servers gracefully."
echo "========================================="

# Wait for background processes
wait $BACKEND_PID
wait $FRONTEND_PID
