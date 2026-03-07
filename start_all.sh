#!/bin/bash

# Dev-Store Main Start Script

echo "================================================"
echo " 🚀 Starting Dev-Store Application"
echo "================================================"
echo ""

# Check if backend is set up
if [ ! -d "backend/venv" ]; then
    echo "[WARNING] Backend not set up. Running setup..."
    cd backend
    chmod +x setup-backend.sh
    ./setup-backend.sh
    cd ..
    echo ""
fi

# Check if frontend is set up
if [ ! -d "frontend/node_modules" ]; then
    echo "[WARNING] Frontend not set up. Running setup..."
    cd frontend
    chmod +x setup-frontend.sh
    ./setup-frontend.sh
    cd ..
    echo ""
fi

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down Dev-Store services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT SIGTERM

# Start backend server
echo "🟢 Starting Backend (FastAPI) on port 8000..."
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 2

# Start frontend server
echo "🟢 Starting Frontend (Next.js) on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "================================================"
echo " ✨ Dev-Store is Running!"
echo "================================================"
echo " 📱 Frontend:  http://localhost:3000"
echo " 🔧 Backend:   http://localhost:8000"
echo " 📚 API Docs:  http://localhost:8000/docs"
echo ""
echo " Press Ctrl+C to stop all servers"
echo "================================================"
echo ""

# Wait for background processes
wait $BACKEND_PID
wait $FRONTEND_PID
