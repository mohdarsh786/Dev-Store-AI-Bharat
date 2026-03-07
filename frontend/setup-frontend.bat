@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

:: ╔══════════════════════════════════════════════════════════════════╗
:: ║          DevStore AI Bharat — Frontend Setup Script             ║
:: ║          Next.js 16 + AWS Bedrock + OpenSearch                  ║
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

:: ─── Step 1: Node.js check ───────────────────────────────────────────────────
echo  [1/4] Checking Node.js...
where node >nul 2>&1
if !errorlevel! neq 0 (
    echo.
    echo  [ERROR] Node.js not found. Install from https://nodejs.org (v18+)
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('node --version') do set NODE_VER=%%v
echo        Node.js !NODE_VER! detected. OK.
echo.

:: ─── Step 2: .env.local check ────────────────────────────────────────────────
echo  [2/4] Checking environment configuration...
if not exist ".env.local" (
    echo.
    echo  ╔══════════════════════════════════════════════════════════════════╗
    echo  ║  [WARNING]  .env.local not found!                               ║
    echo  ║                                                                  ║
    echo  ║  Required frontend config:                                       ║
    echo  ║    • BACKEND_URL=http://localhost:8000                           ║
    echo  ║                                                                  ║
    echo  ║  Create frontend/.env.local before continuing.                   ║
    echo  ║  See README.md § 3 "Security Protocols" for all required vars.  ║
    echo  ╚══════════════════════════════════════════════════════════════════╝
    echo.
    echo  The app will start with demo data only (backend offline mode).
    echo.
    choice /C YN /M "  Continue anyway? (Y=Yes, N=Exit)"
    if !errorlevel! equ 2 exit /b 1
    echo.
) else (
    echo        .env.local found. Secrets loaded server-side only. OK.
    echo.
)

:: ─── Step 3: npm install ─────────────────────────────────────────────────────
echo  [3/4] Installing dependencies...
if not exist "node_modules" (
    echo        node_modules not found. Running npm install...
    echo.
    npm install
    if !errorlevel! neq 0 (
        echo.
        echo  [ERROR] npm install failed. Check your Node.js installation.
        pause
        exit /b 1
    )
    echo.
    echo        Dependencies installed successfully.
) else (
    echo        node_modules exists. Skipping install.
)
echo.

:: ─── Step 4: Launch ──────────────────────────────────────────────────────────
echo  [4/4] Launching Next.js dev server...
echo.
echo  ╔══════════════════════════════════════════════════════════════════╗
echo  ║                                                                  ║
echo  ║   ✅  DevStore Initialized — AI for Bharat                       ║
echo  ║                                                                  ║
echo  ║   Frontend:  http://localhost:3000                               ║
echo  ║   API Routes: http://localhost:3000/api/*                        ║
echo  ║                                                                  ║
echo  ║   Backend (start separately):                                    ║
echo  ║     cd ..\backend                                                ║
echo  ║     uvicorn api_gateway:app --reload --port 8000                 ║
echo  ║                                                                  ║
echo  ║   Press Ctrl+C to stop the server                                ║
echo  ╚══════════════════════════════════════════════════════════════════╝
echo.

npm run dev
