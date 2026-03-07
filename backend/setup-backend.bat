@echo off
setlocal enabledelayedexpansion

echo ================================================
echo  Dev-Store Backend Setup (Windows)
echo ================================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

REM Check Python version
echo Checking Python version...
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python !PYVER! detected.
echo.

REM Create virtual environment
if exist venv\Scripts\activate.bat (
    echo [INFO] Virtual environment already exists.
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
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo.

REM Install dependencies
echo Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] Dependencies installed.
echo.

REM Copy environment template
if not exist .env (
    if exist .env.example (
        echo Creating .env file...
        copy .env.example .env >nul
        echo [OK] .env created. Please edit backend\.env with your AWS credentials.
    ) else (
        echo [WARNING] .env.example not found.
    )
) else (
    echo [INFO] .env already exists.
)
echo.

REM Clean cache
echo Cleaning Python cache...
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
echo [OK] Cache cleaned.
echo.

echo ================================================
echo  Backend Setup Complete!
echo ================================================
echo.
echo Next steps:
echo   1. Edit .env with your AWS credentials
echo   2. Test connections: cd tests ^&^& python test_connections_simple.py
echo   3. Create OpenSearch index: python setup_opensearch_index.py
echo   4. Start server: uvicorn api_gateway:app --reload --port 8000
echo.
echo API docs will be at: http://localhost:8000/api/docs
echo.

pause
