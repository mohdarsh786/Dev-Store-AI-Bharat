@echo off
REM DevStore Frontend Setup Script for Windows

echo Setting up DevStore Frontend...

REM Check Node version
node --version

REM Navigate to frontend directory
cd frontend

REM Install dependencies
echo Installing dependencies...
call npm install

REM Copy environment template
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo Please edit frontend\.env with your configuration
)

echo.
echo Frontend setup complete!
echo.
echo To start the development server, run:
echo   cd frontend ^&^& npm run dev
echo.
echo To build for production, run:
echo   cd frontend ^&^& npm run build
echo.

pause
