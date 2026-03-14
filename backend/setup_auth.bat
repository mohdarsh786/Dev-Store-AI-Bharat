@echo off
echo ========================================
echo DevStore - Authentication Setup
echo ========================================
echo.
echo This script will:
echo 1. Install authentication dependencies
echo 2. Run database migration for users table
echo 3. Test authentication endpoints
echo.
pause

REM Activate virtual environment
call .venv\Scripts\activate

REM Step 1: Install dependencies
echo.
echo [1/3] Installing authentication dependencies...
echo ----------------------------------------
pip install passlib[bcrypt]==1.7.4 python-jose[cryptography]==3.3.0 authlib==1.3.0 itsdangerous==2.1.2 emails==0.6 jinja2==3.1.3
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Step 2: Run migration
echo.
echo [2/3] Running users table migration...
echo ----------------------------------------
python run_migrations.py --file 010_enhanced_users_auth.sql
if errorlevel 1 (
    echo ERROR: Failed to run migration
    pause
    exit /b 1
)

REM Step 3: Test
echo.
echo [3/3] Testing authentication system...
echo ----------------------------------------
echo Starting backend server...
echo.
echo Once server starts, test these endpoints:
echo.
echo POST /api/auth/signup - Create new account
echo POST /api/auth/login - Login
echo POST /api/auth/oauth/google - Google OAuth
echo POST /api/auth/oauth/github - GitHub OAuth
echo POST /api/auth/password/forgot - Request password reset
echo POST /api/auth/password/reset - Reset password
echo POST /api/auth/password/change - Change password
echo GET  /api/auth/me - Get current user
echo.
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.
uvicorn main:app --reload --port 8000
