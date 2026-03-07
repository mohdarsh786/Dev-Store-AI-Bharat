@echo off
echo ========================================
echo DevStore - Database Setup
echo ========================================
echo.
echo This script will:
echo 1. Start PostgreSQL, Redis, OpenSearch (Docker)
echo 2. Run database migrations
echo 3. Populate database with resources
echo.
pause

REM Step 1: Start Docker infrastructure
echo.
echo [1/3] Starting Docker infrastructure...
echo ----------------------------------------
cd ..
docker-compose up -d
if errorlevel 1 (
    echo ERROR: Failed to start Docker containers
    echo Make sure Docker Desktop is installed and running
    pause
    exit /b 1
)

echo Waiting for services to be ready...
timeout /t 10 /nobreak > nul

REM Step 2: Run migrations
echo.
echo [2/3] Running database migrations...
echo ----------------------------------------
cd backend
call .venv\Scripts\activate
python run_migrations.py
if errorlevel 1 (
    echo ERROR: Failed to run migrations
    pause
    exit /b 1
)

REM Step 3: Run ingestion
echo.
echo [3/3] Populating database...
echo ----------------------------------------
echo This will fetch data and populate PostgreSQL...
echo.

python -c "import asyncio; from ingestion.pipeline import run_ingestion; result = asyncio.run(run_ingestion(sources=['github', 'huggingface', 'kaggle', 'openrouter'])); print(f'\nStatus: {result[\"status\"]}'); print(f'Inserted: {result[\"stats\"][\"inserted\"]}'); print(f'Updated: {result[\"stats\"][\"updated\"]}')"

if errorlevel 1 (
    echo ERROR: Ingestion failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✓ Database Setup Complete!
echo ========================================
echo.
echo Your PostgreSQL database is now populated with resources.
echo.
echo Services running:
echo - PostgreSQL: localhost:5432
echo - Redis: localhost:6379
echo - OpenSearch: localhost:9200
echo.
echo To check database:
echo   psql postgresql://devstore:devstore123@localhost:5432/devstore
echo.
echo To start backend with database:
echo   cd backend
echo   .venv\Scripts\activate
echo   uvicorn main:app --reload
echo.
pause
