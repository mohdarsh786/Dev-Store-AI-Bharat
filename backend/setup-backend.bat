@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

:: ╔══════════════════════════════════════════════════════════════════╗
:: ║          DevStore AI Bharat — Backend Setup Script              ║
:: ║          FastAPI + AWS Bedrock + Pinecone + Neon                 ║
:: ╚══════════════════════════════════════════════════════════════════╝

echo.
echo  ██████╗ ███████╗██╗   ██╗    ███████╗████████╗ ██████╗ ██████╗ ███████╗
echo  ██╔══██╗██╔════╝██║   ██║    ██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝
echo  ██║  ██║█████╗  ██║   ██║    ███████╗   ██║   ██║   ██║██████╔╝█████╗
echo  ██║  ██║██╔══╝  ╚██╗ ██╔╝    ╚════██║   ██║   ██║   ██║██╔══██╗██╔══╝
echo  ██████╔╝███████╗ ╚████╔╝     ███████║   ██║   ╚██████╔╝██║  ██║███████╗
echo  ╚═════╝ ╚══════╝  ╚═══╝      ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝
echo.
echo                    AI for Bharat — Developer Marketplace
echo                    ─────────────────────────────────────
echo.

:: ─── Step 1: Python check ───────────────────────────────────────────────────
echo  [1/5] Checking Python...
where python >nul 2>&1
if !errorlevel! neq 0 (
    echo.
    echo  [ERROR] Python not found. Install from https://python.org (v3.11+)
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do set PYTHON_VER=%%v
echo        !PYTHON_VER! detected. OK.
echo.

:: ─── Step 2: .env check ──────────────────────────────────────────────────────
echo  [2/5] Checking environment configuration...
if not exist ".env" (
    echo.
    echo  ╔══════════════════════════════════════════════════════════════════╗
    echo  ║  [WARNING]  .env not found!                                      ║
    echo  ║                                                                  ║
    echo  ║  Required backend config:                                        ║
    echo  ║    • DATABASE_URL=postgresql://user:pass@host/db                 ║
    echo  ║    • PINECONE_API_KEY=your-pinecone-api-key                      ║
    echo  ║    • AWS_ACCESS_KEY_ID=your-aws-access-key                       ║
    echo  ║    • AWS_SECRET_ACCESS_KEY=your-aws-secret-key                   ║
    echo  ║    • BEDROCK_MODEL_ID=your-bedrock-model-arn                     ║
    echo  ║                                                                  ║
    echo  ║  Copy .env.example to .env and configure.                       ║
    echo  ╚══════════════════════════════════════════════════════════════════╝
    echo.
    echo  The server will start but AI features will be disabled.
    echo.
    choice /C YN /M "  Continue anyway? (Y=Yes, N=Exit)"
    if !errorlevel! equ 2 exit /b 1
    echo.
) else (
    echo        .env found. Configuration loaded. OK.
    echo.
)

:: ─── Step 3: Virtual environment ────────────────────────────────────────────
echo  [3/5] Setting up virtual environment...
if not exist "venv" (
    echo        Creating virtual environment...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo.
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo        Virtual environment created.
) else (
    echo        Virtual environment exists. Skipping creation.
)
echo.

:: ─── Step 4: Install dependencies ───────────────────────────────────────────
echo  [4/5] Installing Python dependencies...
call venv\Scripts\activate.bat
if !errorlevel! neq 0 (
    echo.
    echo  [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo.
    echo  [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo        Dependencies installed successfully.
echo.

:: ─── Step 5: Launch ──────────────────────────────────────────────────────────
echo  [5/5] Launching FastAPI dev server...
echo.
echo  ╔══════════════════════════════════════════════════════════════════╗
echo  ║                                                                  ║
echo  ║   ✅  DevStore Backend Initialized — AI for Bharat               ║
echo  ║                                                                  ║
echo  ║   Backend API:  http://localhost:8000                            ║
echo  ║   API Docs:     http://localhost:8000/docs                       ║
echo  ║   Health Check: http://localhost:8000/api/v1/health              ║
echo  ║                                                                  ║
echo  ║   Frontend (start separately):                                   ║
echo  ║     cd ..\frontend                                               ║
echo  ║     npm install                                                  ║
echo  ║     npm run dev                                                  ║
echo  ║                                                                  ║
echo  ║   Press Ctrl+C to stop the server                                ║
echo  ╚══════════════════════════════════════════════════════════════════╝
echo.

uvicorn main:app --reload --port 8000