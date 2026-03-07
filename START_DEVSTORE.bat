@echo off
echo ================================================
echo  Starting Dev-Store Application
echo ================================================
echo.

REM Check if backend is set up
if not exist backend\venv (
    echo [WARNING] Backend not set up. Running setup...
    echo.
    cd backend
    call setup-backend.bat
    cd ..
    echo.
)

REM Check if frontend is set up
if not exist frontend\node_modules (
    echo [WARNING] Frontend not set up. Running setup...
    echo.
    cd frontend
    call setup-frontend.bat
    cd ..
    echo.
)

echo Starting Backend Server...
start "Dev-Store Backend" cmd /k "cd backend && venv\Scripts\activate.bat && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak >nul

echo Starting Frontend Server...
start "Dev-Store Frontend" cmd /k "cd frontend && npm run dev"
timeout /t 3 /nobreak >nul

echo.
echo ================================================
echo  Dev-Store is Starting!
echo ================================================
echo.
echo Backend API:  http://localhost:8000
echo API Docs:     http://localhost:8000/docs
echo Frontend:     http://localhost:3000
echo.
echo Servers are running in separate windows.
echo Close those windows to stop the servers.
echo.
echo Press any key to open browser...
pause >nul

start http://localhost:3000

echo.
pause
