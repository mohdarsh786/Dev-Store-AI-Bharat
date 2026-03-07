@echo off
echo ========================================
echo DevStore - AWS Database Setup
echo ========================================
echo.
echo This script will:
echo 1. Connect to your AWS RDS PostgreSQL
echo 2. Run database migrations
echo 3. Fetch data and populate database
echo 4. Configure backend to use AWS database
echo.
echo IMPORTANT: Make sure your .env file has the correct AWS RDS endpoint
echo Example: DATABASE_URL=postgresql://username:password@your-rds-endpoint.rds.amazonaws.com:5432/devstore
echo.
pause

REM Activate virtual environment
call .venv\Scripts\activate

REM Step 1: Test database connection
echo.
echo [1/3] Testing AWS RDS connection...
echo ----------------------------------------
python -c "from clients.database import DatabaseClient; import asyncio; client = DatabaseClient(); asyncio.run(client.health_check()); print('✓ Database connection successful')"
if errorlevel 1 (
    echo.
    echo ERROR: Cannot connect to AWS RDS database
    echo.
    echo Please check:
    echo 1. DATABASE_URL in .env file is correct
    echo 2. RDS security group allows your IP
    echo 3. RDS instance is running
    echo 4. Database credentials are correct
    echo.
    pause
    exit /b 1
)

REM Step 2: Run migrations
echo.
echo [2/3] Running database migrations...
echo ----------------------------------------
python run_migrations.py
if errorlevel 1 (
    echo ERROR: Failed to run migrations
    pause
    exit /b 1
)

REM Step 3: Populate database
echo.
echo [3/3] Fetching data and populating AWS RDS...
echo ----------------------------------------
echo This will:
echo - Fetch from GitHub, HuggingFace, Kaggle, OpenRouter
echo - Save to your AWS RDS PostgreSQL
echo - Generate embeddings using AWS Bedrock
echo - Index in AWS OpenSearch
echo.
echo This may take 5-10 minutes...
echo.

python -c "import asyncio; from ingestion.pipeline import run_ingestion; result = asyncio.run(run_ingestion(sources=['github', 'huggingface', 'kaggle', 'openrouter'])); print(f'\n✓ Ingestion complete!'); print(f'Inserted: {result[\"stats\"][\"inserted\"]}'); print(f'Updated: {result[\"stats\"][\"updated\"]}'); print(f'Failed: {result[\"stats\"][\"failed\"]}')"

if errorlevel 1 (
    echo.
    echo ERROR: Ingestion failed
    echo Check the logs above for details
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✓ AWS Database Setup Complete!
echo ========================================
echo.
echo Your AWS RDS PostgreSQL is now populated with resources.
echo.
echo Backend will now automatically use AWS database instead of JSON files.
echo.
echo To start backend:
echo   uvicorn main:app --reload
echo.
echo To verify data:
echo   python -c "from ingestion.repository import IngestionRepository; from clients.database import DatabaseClient; repo = IngestionRepository(DatabaseClient()); stats = repo.get_resource_stats(); print(stats)"
echo.
pause
