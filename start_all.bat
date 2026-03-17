@echo off
setlocal EnableDelayedExpansion

:: Dev-Store Main Start Script

echo ================================================
echo  🚀 Starting Dev-Store Application
echo ================================================
echo.

:: Check if backend is set up
if not exist "backend\venv" (
    echo [WARNING] Backend not set up. Running setup...
    cd backend
    call setup-backend.bat
    cd ..
    echo.
)

:: Check if frontend is set up
if not exist "frontend\node_modules" (
    echo [WARNING] Frontend not set up. Running setup...
    cd frontend
    call setup-frontend.bat
    cd ..
    echo.
)

echo ================================================
echo  ✨ Dev-Store Setup Complete!
echo ================================================
echo  📱 Frontend:  http://localhost:3000
echo  🔧 Backend:   http://localhost:8000
echo  📚 API Docs:  http://localhost:8000/docs
echo ================================================
echo.

echo 🟢 Starting Backend (FastAPI)...
start "Dev-Store Backend" /D "backend" cmd /c "venv\Scripts\activate && uvicorn main:app --reload --port 8000"

:: Wait a moment for backend to initialize
timeout /t 3 /nobreak > nul

echo 🟢 Starting Frontend (Next.js)...
start "Dev-Store Frontend" /D "frontend" cmd /c "npm run dev"

echo.
echo ================================================
echo  🚀 Both servers are starting in separate windows.
echo  Close those windows or press Ctrl+C in them to stop.
echo ================================================
echo.

pause