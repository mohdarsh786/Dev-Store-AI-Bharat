@echo off
REM DevStore Backend Setup Script for Windows

echo Setting up DevStore Backend...

REM Check Python version
python --version

REM Navigate to backend directory
cd backend

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Copy environment template
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo Please edit backend\.env with your configuration
)

echo.
echo Backend setup complete!
echo.
echo To activate the virtual environment, run:
echo   cd backend ^&^& venv\Scripts\activate.bat
echo.
echo To start the development server, run:
echo   uvicorn main:app --reload --port 8000
echo.

pause
