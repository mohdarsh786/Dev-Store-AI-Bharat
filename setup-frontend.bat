@echo off
setlocal enabledelayedexpansion

echo ================================================
echo  DevStore Frontend Setup for Windows
echo ================================================
echo.

REM Check if Node.js is installed
where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js not found. Please install Node.js 20 LTS from https://nodejs.org
    pause
    exit /b 1
)

REM Check Node version (require 18+)
echo Checking Node.js version...
for /f "tokens=1" %%v in ('node --version') do set NODEVER=%%v
set NODEMAJOR=!NODEVER:~1!
for /f "tokens=1 delims=." %%a in ("!NODEMAJOR!") do set NODEMAJOR=%%a
if !NODEMAJOR! LSS 18 (
    echo [ERROR] Node.js 18+ required. Found: !NODEVER!. Please upgrade to Node.js 20 LTS.
    pause
    exit /b 1
)
echo [OK] Node.js !NODEVER! detected.

REM Check npm
where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm not found. Please reinstall Node.js.
    pause
    exit /b 1
)
echo [OK] npm detected.
echo.

REM Navigate to frontend directory relative to this script
cd /d "%~dp0frontend"
if errorlevel 1 (
    echo [ERROR] Could not navigate to frontend directory.
    pause
    exit /b 1
)

REM Install dependencies
echo Installing frontend dependencies...
call npm install
if errorlevel 1 (
    echo [ERROR] Failed to install frontend dependencies.
    pause
    exit /b 1
)
echo [OK] Dependencies installed successfully.
echo.

REM Create single environment file
if not exist .env (
    echo Creating frontend\.env...
    (
        echo # DevStore Frontend Environment
        echo VITE_API_URL=http://localhost:8000
        echo VITE_API_BASE_URL=http://localhost:8000/api/v1
        echo VITE_API_TIMEOUT=30000
        echo VITE_ENABLE_AUTH=false
        echo VITE_ENABLE_ANALYTICS=false
        echo VITE_ENVIRONMENT=development
    ) > .env
    echo [OK] .env created. Please edit frontend\.env with your deployed API endpoint.
) else (
    echo [INFO] .env already exists. Skipping.
)
echo.

REM Run a quick audit check (non-blocking)
echo Running npm audit (informational)...
call npm audit --audit-level=high 2>nul
echo.

echo ================================================
echo  Frontend setup complete!
echo ================================================
echo.
echo Next steps:
echo   1. Edit frontend\.env with your API endpoint (VITE_API_URL)
echo   2. Start the development server:
echo      cd frontend ^&^& npm run dev
echo   3. Build for production:
echo      cd frontend ^&^& npm run build
echo   4. Run tests:
echo      cd frontend ^&^& npm test
echo.

pause
