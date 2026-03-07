@echo off
echo ========================================
echo DevStore - Starting Development Environment
echo ========================================
echo.

REM Check if backend virtual environment exists
if not exist "backend\.venv" (
    echo Creating Python virtual environment...
    cd backend
    python -m venv .venv
    call .venv\Scripts\activate
    pip install -r requirements.txt
    cd ..
)

REM Check if frontend node_modules exists
if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
)

echo.
echo Starting services...
echo.
echo [1/2] Starting Backend API on http://localhost:8000
start "DevStore Backend" cmd /k "cd backend && .venv\Scripts\activate && uvicorn main:app --reload --port 8000"

timeout /t 3 /nobreak > nul

echo [2/2] Starting Frontend on http://localhost:3000
start "DevStore Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo ✓ DevStore is starting!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to open browser...
pause > nul

start http://localhost:3000

echo.
echo To stop: Close the terminal windows or press Ctrl+C
echo.
