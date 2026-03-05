@echo off
echo ============================================================
echo DEV STORE INGESTION PIPELINE TEST
echo ============================================================
echo.

cd /d "%~dp0"

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo ============================================================
echo TESTING HTTP FETCHERS
echo ============================================================
echo.

python ingestion\test_http_fetchers.py

echo.
echo ============================================================
echo TEST COMPLETE
echo ============================================================
echo.
echo To run full ingestion:
echo   python ingestion\run_ingestion.py
echo.

pause
