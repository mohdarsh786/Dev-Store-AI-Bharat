@echo off
echo ============================================================
echo DEV STORE FULL INGESTION PIPELINE
echo ============================================================
echo.

cd /d "%~dp0"

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo ============================================================
echo RUNNING FULL INGESTION
echo ============================================================
echo.
echo This will fetch data from:
echo   - HuggingFace API (HTTP)
echo   - OpenRouter API (HTTP)
echo   - GitHub API (HTTP)
echo   - RapidAPI (Scrapy crawler)
echo.

python ingestion\run_ingestion.py

echo.
echo ============================================================
echo INGESTION COMPLETE
echo ============================================================
echo.
echo Check output files in: ingestion\output\
echo.

pause
