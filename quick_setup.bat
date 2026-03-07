@echo off
echo ========================================
echo DevStore - Quick Setup Script
echo ========================================
echo.
echo This script will set up your development environment.
echo.
pause

REM Step 1: Backend Setup
echo.
echo [Step 1/4] Setting up Backend...
echo ----------------------------------------
cd backend

if not exist ".venv" (
    echo Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Make sure Python 3.11+ is installed
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call .venv\Scripts\activate

echo Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo ✓ Backend setup complete
cd ..

REM Step 2: Frontend Setup
echo.
echo [Step 2/4] Setting up Frontend...
echo ----------------------------------------
cd frontend

if not exist "node_modules" (
    echo Installing Node.js dependencies...
    call npm install
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        echo Make sure Node.js is installed
        pause
        exit /b 1
    )
)

echo ✓ Frontend setup complete
cd ..

REM Step 3: Environment Configuration
echo.
echo [Step 3/4] Configuring Environment...
echo ----------------------------------------

if not exist "backend\.env" (
    echo Creating backend .env file...
    copy backend\.env.example backend\.env
    echo ✓ Created backend/.env
) else (
    echo ✓ backend/.env already exists
)

if not exist "frontend\.env.local" (
    echo Creating frontend .env.local file...
    echo BACKEND_URL=http://localhost:8000 > frontend\.env.local
    echo ✓ Created frontend/.env.local
) else (
    echo ✓ frontend/.env.local already exists
)

REM Step 4: Verify Setup
echo.
echo [Step 4/4] Verifying Setup...
echo ----------------------------------------

echo Checking Python...
python --version
if errorlevel 1 (
    echo WARNING: Python not found in PATH
)

echo Checking Node.js...
node --version
if errorlevel 1 (
    echo WARNING: Node.js not found in PATH
)

echo Checking npm...
npm --version
if errorlevel 1 (
    echo WARNING: npm not found in PATH
)

echo.
echo ========================================
echo ✓ Setup Complete!
echo ========================================
echo.
echo Next steps:
echo.
echo 1. Start the development environment:
echo    ^> start_dev.bat
echo.
echo 2. Or start services manually:
echo    Backend:  cd backend ^&^& .venv\Scripts\activate ^&^& uvicorn main:app --reload
echo    Frontend: cd frontend ^&^& npm run dev
echo.
echo 3. Open browser: http://localhost:3000
echo.
echo Optional: Set up local infrastructure (PostgreSQL, Redis, OpenSearch)
echo    ^> docker-compose up -d
echo.
echo For full setup guide, see: COMPLETE_SETUP_GUIDE.md
echo.
pause
