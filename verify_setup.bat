@echo off
echo ========================================
echo DevStore - Setup Verification
echo ========================================
echo.

echo [1/6] Checking Python...
python --version
if errorlevel 1 (
    echo ✗ Python not found
    goto :error
) else (
    echo ✓ Python installed
)

echo.
echo [2/6] Checking Node.js...
node --version
if errorlevel 1 (
    echo ✗ Node.js not found
    goto :error
) else (
    echo ✓ Node.js installed
)

echo.
echo [3/6] Checking Backend Virtual Environment...
if exist "backend\.venv" (
    echo ✓ Virtual environment exists
) else (
    echo ✗ Virtual environment not found
    goto :error
)

echo.
echo [4/6] Checking Frontend Dependencies...
if exist "frontend\node_modules" (
    echo ✓ Frontend dependencies installed
) else (
    echo ✗ Frontend dependencies not installed
    goto :error
)

echo.
echo [5/6] Checking Data Files...
if exist "backend\ingestion\output\models.json" (
    echo ✓ models.json found
) else (
    echo ✗ models.json not found
)
if exist "backend\ingestion\output\github_resources.json" (
    echo ✓ github_resources.json found
) else (
    echo ✗ github_resources.json not found
)
if exist "backend\ingestion\output\huggingface_datasets.json" (
    echo ✓ huggingface_datasets.json found
) else (
    echo ✗ huggingface_datasets.json not found
)
if exist "backend\ingestion\output\kaggle_datasets.json" (
    echo ✓ kaggle_datasets.json found
) else (
    echo ✗ kaggle_datasets.json not found
)

echo.
echo [6/6] Testing Backend Imports...
cd backend
.venv\Scripts\python.exe -c "from ingestion.pipeline import run_ingestion; print('✓ Pipeline imports work')"
if errorlevel 1 (
    cd ..
    echo ✗ Import test failed
    goto :error
)
cd ..

echo.
echo ========================================
echo ✓ All Checks Passed!
echo ========================================
echo.
echo Your system is ready to run DevStore.
echo.
echo Next steps:
echo   1. Run: start_dev.bat
echo   2. Open: http://localhost:3000
echo.
pause
exit /b 0

:error
echo.
echo ========================================
echo ✗ Setup Incomplete
echo ========================================
echo.
echo Please run: quick_setup.bat
echo.
pause
exit /b 1
