@echo off
setlocal enabledelayedexpansion

echo ================================================
echo  DevStore Backend Setup for Windows
echo ================================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

REM Check Python version (require 3.11+)
echo Checking Python version...
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)
if !PYMAJOR! LSS 3 (
    echo [ERROR] Python 3.11+ required. Found: !PYVER!
    pause
    exit /b 1
)
if !PYMAJOR! EQU 3 if !PYMINOR! LSS 11 (
    echo [WARNING] Python 3.11+ recommended. Found: !PYVER!
)
echo [OK] Python !PYVER! detected.
echo.

REM Navigate to backend directory relative to this script
cd /d "%~dp0backend"
if errorlevel 1 (
    echo [ERROR] Could not navigate to backend directory.
    pause
    exit /b 1
)

REM Create virtual environment (skip if already exists)
if exist venv\Scripts\activate.bat (
    echo [INFO] Virtual environment already exists. Skipping creation.
) else (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.
echo.

REM Upgrade pip and build tools
echo Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [WARNING] Build tools upgrade failed. Continuing with existing versions.
)
echo.

REM Install dependencies
echo Installing backend dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies. Check requirements.txt and your Python version.
    pause
    exit /b 1
)
echo [OK] Dependencies installed successfully.
echo.

REM Copy environment template
if not exist .env (
    if exist .env.example (
        echo Creating .env file from template...
        copy .env.example .env
        echo [OK] .env created. Please edit backend\.env with your AWS/DB configuration.
    ) else (
        echo [WARNING] .env.example not found. Please create backend\.env manually.
    )
) else (
    echo [INFO] .env already exists. Skipping.
)
echo.

echo ================================================
echo  Backend setup complete!
echo ================================================
echo.
echo Next steps:
echo   1. Edit backend\.env with your AWS, DB, and Redis config
echo   2. Run database migrations:
echo      cd backend ^&^& venv\Scripts\activate.bat ^&^& python run_migrations.py
echo   3. Start the development server:
echo      cd backend ^&^& venv\Scripts\activate.bat ^&^& uvicorn main:app --reload --port 8000
echo.

pause
