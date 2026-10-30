# Dev-Store

<p align="center">
  <h3 align="center">A developer resource discovery platform combining PostgreSQL metadata filtering, Pinecone vector search, and Amazon Bedrock conversational retrieval.</h3>
</p>

<p align="center">
  <a href="https://main.d2goix7t3vcz9c.amplifyapp.com"><img src="https://img.shields.io/badge/TRY%20LIVE%20DEMO-Launch%20Dev--Store-FF9900?style=for-the-badge&logo=aws-amplify&logoColor=white" alt="Live Demo"/></a>
  &nbsp;
  <a href="https://dev-store-ai-bharat.onrender.com"><img src="https://img.shields.io/badge/Backend-Render-46E3B7?style=for-the-badge&logo=render" alt="Backend"/></a>
  &nbsp;
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache--2.0-yellow.svg?style=for-the-badge" alt="License"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-16.1.6-000000?style=for-the-badge&logo=next.js&logoColor=white" alt="Next.js"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115.6-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PostgreSQL-Neon-00E599?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"/>
  <img src="https://img.shields.io/badge/Pinecone-6.0.0-000000?style=for-the-badge&logo=pinecone&logoColor=white" alt="Pinecone"/>
  <img src="https://img.shields.io/badge/AWS%20Bedrock-Titan%20%2F%20Nova%20Micro-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="AWS Bedrock"/>
</p>

## About

Dev-Store is a discovery platform for developers to search, evaluate, and integrate APIs, machine learning models, and datasets. The application combines relational SQL filtering in PostgreSQL with vector similarity search in Pinecone and embedding generation via Amazon Bedrock. It provides a web dashboard with search filters, ranking scores, copyable installation commands, and an integrated conversational retrieval assistant.

## Platform Metrics

Resource counts from bundled JSON files in `backend/`:
* GitHub repositories: 654 records (`backend/github_resources.json`)
* Hugging Face datasets: 100 records (`backend/huggingface_datasets.json`)
* Kaggle datasets: 100 records (`backend/kaggle_datasets.json`)
* Models: 1,403 records total across Hugging Face (1,081) and OpenRouter (322) (`backend/models.json`)
* Total bundled resource records: 2,257

Test counts collected per test file (`pytest --collect-only -q`):
* `backend/tests/test_models.py`: 16 tests
* `backend/tests/test_database_client.py`: 14 tests
* `backend/tests/test_ranking_features.py`: 12 tests
* `backend/tests/test_models_validation.py`: 11 tests
* `backend/tests/test_models_basic.py`: 8 tests
* `backend/tests/test_bedrock.py`: 2 tests
* `backend/tests/test_brain.py`: 1 test
* `backend/tests/test_intent_search.py`: 1 test
* `backend/tests/test_rag_system.py`: 1 test
* Total successfully collected tests: 66 tests across 9 test files
* Total tests executed in CI workflow (`.github/workflows/ci-cd.yml`): 47 tests across `test_models.py`, `test_models_basic.py`, `test_models_validation.py`, and `test_ranking_features.py`

## Features

* Dual-layer search queries PostgreSQL for keyword and metadata filtering, and Pinecone for vector semantic similarity.
* Exact keyword matches in PostgreSQL receive an additive score boost of 2.0 before rank normalization.
* Composite ranking scores weight semantic relevance at 40 percent, popularity at 30 percent, optimization at 20 percent, and freshness at 10 percent.
* Popularity scores normalize GitHub star counts against 100,000, downloads against 1,000,000, and active users against 10,000.
* Optimization scores evaluate request latency against a 1,000 millisecond ceiling, cost per request against 0.01 dollars, and documentation availability.
* Freshness scores combine 365-day linear update recency with operational health status values.
* Trending scores calculate short-term velocity based on recent downloads, views, bookmarks, and growth rates.
* Multi-source ingestion harvests technical metadata from GitHub, Hugging Face, Kaggle, and OpenRouter public APIs.
* Pipeline deduplication computes an in-memory unique key per source and normalized URL.
* PostgreSQL upsert logic evaluates SHA-256 content and embedding hashes to bypass redundant Bedrock embedding calls when descriptions have not changed.
* Backfill scripts embed text with Amazon Bedrock Titan and upload vectors to Pinecone in batches of 50 with exponential backoff on throttling.
* Conversational assistant retrieves top 5 context chunks from Pinecone and generates answers using Amazon Nova Micro via AWS Bedrock runtime.
* Next.js server-side rewrites route browser API calls through a reverse proxy to prevent mixed-content blocks when calling an HTTP backend from an HTTPS domain.
* Frontend authentication supports Google and GitHub OAuth providers alongside mock credential verification.
* In-memory response caching uses TTLCache with a 300-second time-to-live to cache API gateway responses.
* Ingestion process execution is guarded against overlapping runs on a single worker using an in-memory asyncio lock.
* Multilingual queries in English, Hindi, and Hinglish are handled by the foundation model and embedding space, supported by a frontend UI toggle and fallback string.

## Tech Stack

### Frontend
![Next.js](https://img.shields.io/badge/Next.js-16.1.6-000000?style=for-the-badge&logo=next.js&logoColor=white)
![React](https://img.shields.io/badge/React-19.2.3-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
![NextAuth](https://img.shields.io/badge/NextAuth-5.0.0--beta.30-purple?style=for-the-badge&logo=auth0&logoColor=white)
![SWR](https://img.shields.io/badge/SWR-2.4.1-black?style=for-the-badge&logo=vercel&logoColor=white)
![Axios](https://img.shields.io/badge/Axios-1.13.6-5A29E4?style=for-the-badge&logo=axios&logoColor=white)
![React Flow](https://img.shields.io/badge/React_Flow-11.11.4-FF0072?style=for-the-badge&logo=react&logoColor=white)

### Backend
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-0.34.0-499848?style=for-the-badge&logo=gunicorn&logoColor=white)
![Gunicorn](https://img.shields.io/badge/Gunicorn-23.0.0-499848?style=for-the-badge&logo=gunicorn&logoColor=white)
![Mangum](https://img.shields.io/badge/Mangum-0.19.0-FF9900?style=for-the-badge&logo=aws-lambda&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic_Settings-2.13.1-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![Cachetools](https://img.shields.io/badge/Cachetools-5.3.3-blue?style=for-the-badge)
![HTTPX](https://img.shields.io/badge/HTTPX-0.28.1-111111?style=for-the-badge)
![Aiohttp](https://img.shields.io/badge/Aiohttp-3.11.12-2C5BB4?style=for-the-badge&logo=python&logoColor=white)
![Scrapy](https://img.shields.io/badge/Scrapy-2.12.0-60A839?style=for-the-badge&logo=scrapy&logoColor=white)
![Kaggle API](https://img.shields.io/badge/Kaggle_API-1.6.17-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

### AI and Machine Learning
![AWS Bedrock](https://img.shields.io/badge/AWS_Bedrock-Runtime-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Amazon Titan](https://img.shields.io/badge/Amazon_Titan-Embed_Text_v1-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Amazon Nova Micro](https://img.shields.io/badge/Amazon_Nova_Micro-apac.amazon.nova--micro--v1:0-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)

### Database
![Neon PostgreSQL](https://img.shields.io/badge/Neon_PostgreSQL-Serverless-00E599?style=for-the-badge&logo=postgresql&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-6.0.0-000000?style=for-the-badge&logo=pinecone&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0.20-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)
![Alembic](https://img.shields.io/badge/Alembic-1.12.0-blue?style=for-the-badge)

### Cloud and Deployment
![AWS Amplify](https://img.shields.io/badge/AWS_Amplify-Frontend_Hosting-FF9900?style=for-the-badge&logo=aws-amplify&logoColor=white)
![Render](https://img.shields.io/badge/Render-Backend_Service-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![AWS EC2](https://img.shields.io/badge/AWS_EC2-Amazon_Linux_2023-FF9900?style=for-the-badge&logo=amazon-ec2&logoColor=white)

### Testing and CI
![Pytest](https://img.shields.io/badge/Pytest-8.3.4-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![Hypothesis](https://img.shields.io/badge/Hypothesis-6.130.6-blue?style=for-the-badge)
![Flake8](https://img.shields.io/badge/Flake8-Lint-4B8BBE?style=for-the-badge&logo=python&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

## Architecture

```text
+-----------------------------------------------------------------+
|                         Browser Client                          |
+-----------------------------------------------------------------+
                                |
             HTTPS Requests     | (SWR Client Caching)
                                v
+-----------------------------------------------------------------+
|                       Next.js 16 Frontend                       |
|                      (AWS Amplify Hosting)                      |
|  - DevStoreDashboard                                            |
|  - Next.js Rewrites (Reverse Proxy /api/v1/* to Backend)        |
+-----------------------------------------------------------------+
                                |
         Reverse Proxy Request  |
                                v
+-----------------------------------------------------------------+
|                        FastAPI Backend                          |
|                     (EC2 Gunicorn / Render)                     |
|  - Routers: Search, Resources, RAG, Auth, Health                |
|  - In-Memory TTLCache (300s TTL)                                |
+-----------------------------------------------------------------+
          /                     |                     \
         /                      |                      \
  1. SQL ILIKE & Filters        | 2. Vectorize Query   | 3. Vector Top-40
       /                        |                      \
      v                         v                       v
+------------------+   +------------------+   +-------------------+
| Neon PostgreSQL  |   |   AWS Bedrock    |   | Pinecone Vector   |
| (Serverless DB)  |   | (Titan Embeddings|   | (Serverless Index |
| - Metadata & SQL |   |  + Nova Micro)   |   |  top_k=40)        |
+------------------+   +------------------+   +-------------------+
```

### Search Request Flow

1. The client browser dispatches a search request with query text and optional filters (`category`, `pricing_type`) to the Next.js frontend.
2. Next.js server-side rewrites route the request to the FastAPI backend.
3. The search service initiates two operations:
   - A SQL query on Neon PostgreSQL evaluates keyword matching (`ILIKE` on `name` and `type`) and applies relational filters (`type`, `pricing_type`), applying an additive +2.0 score boost to keyword hits.
   - An AWS Bedrock runtime call converts the query text into a 1536-dimensional vector embedding using Amazon Titan Embeddings (`amazon.titan-embed-text-v1`).
4. The embedding queries Pinecone with `top_k=40` alongside metadata filters on `type` and `is_free`.
5. SQL and Pinecone matches are merged and deduplicated by resource ID.
6. The ranking service calculates composite scores based on relevance, popularity, optimization, and freshness.
7. If the highest score in the batch is below 0.4, an empty list is returned; otherwise, scores are normalized to a 0.99 ceiling.
8. Any SQL matches missing from Pinecone trigger a background daemon thread that embeds and loads them to Pinecone.
9. Results are returned as JSON to the frontend.

### Chat Request Flow

1. The user submits a question and optional filters to `POST /api/v1/rag/chat`.
2. The query is converted into a vector embedding using Amazon Titan via `BedrockClient`.
3. Pinecone is queried with the vector to retrieve the top 5 context chunks (`top_k=5`) with metadata.
4. Context texts and citations (name, category, stars, downloads, score) are extracted into a context block.
5. AWS Bedrock runtime is invoked using Amazon Nova Micro (`apac.amazon.nova-micro-v1:0`) with temperature 0.1 and maxTokens 1024, passing system instructions, context, and user query.
6. The generated Markdown answer and citation sources are returned to the client.
7. If Bedrock invocation fails, the top retrieved Pinecone context chunk text is returned directly as a fallback response.

### Ingestion Pipeline Stages

The ingestion pipeline executes stages in this sequence:
1. Fetch Stage: downloads raw records asynchronously from GitHub, Hugging Face, Kaggle, and OpenRouter APIs.
2. Normalize Stage: maps raw records to the `CanonicalResource` Pydantic model.
3. Deduplicate Stage: filters records using an in-memory dictionary on `source:normalized_url`.
4. Upsert Stage: writes records to Neon PostgreSQL, comparing SHA-256 content and embedding hashes to identify new or changed items.
5. Embed Stage: generates 1536-dimensional embeddings for modified resources using Amazon Bedrock Titan.
6. Index Stage: uploads vectors and metadata to Pinecone in batches of 50.
7. Refresh Rankings: recalculates `rank_score`, `trending_score`, and category rank positions across records.
8. Cache Clear: flushes the in-memory TTLCache.

### Prerequisites

* Python 3.11 (specified in `render.yaml` and `.github/workflows/ci-cd.yml`)
* Node.js 20 or higher (from `@types/node` in `frontend/package.json`)
* AWS Account with Bedrock runtime access (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`)
* Neon Serverless PostgreSQL database URL (`DATABASE_URL`)
* Pinecone account and serverless index (`PINECONE_API_KEY`, `PINECONE_INDEX_NAME`)
* NextAuth secret (`AUTH_SECRET`)
* Optional: GitHub personal access token (`INGESTION_GITHUB_API_TOKEN`) and Kaggle API credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mohdarsh786/Dev-Store-AI-Bharat.git
cd Dev-Store-AI-Bharat
```

2. Install backend dependencies:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
cd ..
```

4. Configure environment files:
```bash
cp backend/.env.example backend/.env
cp frontend/.env.local.example frontend/.env.local
```

5. Run database migrations:
```bash
cd backend
python run_migrations.py
cd ..
```

6. Start development servers:
```bash
# Terminal 1 - Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev

# Or use the root startup script
./start_all.sh  # On Windows: .\start_all.bat
```

### Environment Variables

| Variable Name | Component | Description | Required |
|---|---|---|---|
| `DATABASE_URL` | Backend | PostgreSQL connection URL with SSL mode | Yes |
| `NEON_DATABASE_URL` | Backend | Dedicated connection string for Neon PostgreSQL | No |
| `DB_POOL_SIZE` | Backend | Connection pool base size (defaults to 20) | No |
| `DB_MAX_OVERFLOW` | Backend | Connection pool max overflow (defaults to 10) | No |
| `AWS_REGION` | Backend | Primary AWS region for client initialization (defaults to us-east-1) | Yes |
| `AWS_ACCOUNT_ID` | Backend | AWS account identifier | No |
| `AWS_ACCESS_KEY_ID` | Backend | IAM access key ID for AWS SDK calls | Yes |
| `AWS_SECRET_ACCESS_KEY` | Backend | IAM secret access key for AWS SDK calls | Yes |
| `BEDROCK_REGION` | Backend | AWS region hosting Bedrock runtime (defaults to ap-northeast-1) | No |
| `BEDROCK_MODEL_ID` | Backend | Bedrock model ID or inference profile ARN for text generation | No |
| `EMBEDDING_MODEL_ID` | Backend | Bedrock embedding model ID read directly in client (defaults to Titan v1) | No |
| `BEDROCK_EMBEDDING_MODEL_ID` | Backend | Bedrock embedding model ID read via Settings (defaults to Titan v2) | No |
| `BEDROCK_CLAUDE_ARN` | Backend | Optional ARN for Claude inference profile | No |
| `SYSTEM_PROMPT` | Backend | System prompt passed to Bedrock model invocation | No |
| `PINECONE_API_KEY` | Backend | API key for Pinecone vector database | Yes |
| `PINECONE_INDEX_NAME` | Backend | Target Pinecone vector index (defaults to `devstore_resources`) | Yes |
| `PINECONE_ENVIRONMENT` | Backend | Environment or cloud region descriptor for Pinecone | No |
| `INGESTION_GITHUB_API_TOKEN` | Backend | GitHub personal access token to avoid API rate limits | No |
| `INGESTION_SQS_QUEUE_URL` | Backend | Optional SQS queue URL for ingestion scaling | No |
| `INGESTION_LOCK_KEY` | Backend | Key identifier for ingestion execution lock | No |
| `INGESTION_LOCK_TTL_SECONDS` | Backend | Timeout in seconds for ingestion process lock | No |
| `INGESTION_STATUS_TTL_SECONDS` | Backend | Time to keep latest ingestion status metadata | No |
| `INGESTION_SCHEDULE_HOURS` | Backend | Frequency in hours between scheduled pipeline runs | No |
| `RANKING_SCHEDULE_HOUR_UTC` | Backend | Hour of day (UTC) to trigger ranking recalculations | No |
| `CRAWLER_SNAPSHOT_PREFIX` | Backend | Prefix for dataset backup snapshots | No |
| `API_RATE_LIMIT` | Backend | Maximum permitted API requests per minute | No |
| `API_TIMEOUT` | Backend | Maximum timeout in seconds for API execution | No |
| `CORS_ORIGINS` | Backend | List of permitted CORS origins (defaults to `["*"]`) | No |
| `SECRET_KEY` | Backend | Symmetric secret key for signing JWT tokens | Yes |
| `JWT_ALGORITHM` | Backend | Cryptographic algorithm for JWT signing (defaults to HS256) | No |
| `JWT_EXPIRATION_MINUTES` | Backend | Token validity duration in minutes | No |
| `ENVIRONMENT` | Backend | Execution environment mode (defaults to development) | No |
| `LOG_LEVEL` | Backend | Application logging level (defaults to INFO) | No |
| `SERVER_HOST` | Backend | Binding interface address (defaults to 0.0.0.0) | No |
| `SERVER_PORT` | Backend | Listening port number (defaults to 8000) | No |
| `WORKERS` | Backend | Number of Gunicorn worker processes (defaults to 4) | No |
| `KAGGLE_USERNAME` | Backend | Kaggle username for Kaggle dataset API client | No |
| `KAGGLE_KEY` | Backend | Kaggle API key for Kaggle dataset API client | No |
| `BACKEND_URL` | Frontend | Target backend API URL used by server routes and rewrites | No |
| `NEXT_PUBLIC_API_URL` | Frontend | Public API endpoint URL used in client fetch calls | No |
| `AUTH_SECRET` | Frontend | Encryption secret key for NextAuth session cookies | Yes |
| `AUTH_TRUST_HOST` | Frontend | Instructs NextAuth to trust proxy host headers | No |
| `AUTH_URL` / `NEXTAUTH_URL` | Frontend | Canonical public URL of the frontend application | No |
| `AUTH_GITHUB_ID` | Frontend | GitHub OAuth application client ID | No |
| `AUTH_GITHUB_SECRET` | Frontend | GitHub OAuth application client secret | No |
| `AUTH_GOOGLE_ID` | Frontend | Google Cloud OAuth application client ID | No |
| `AUTH_GOOGLE_SECRET` | Frontend | Google Cloud OAuth application client secret | No |
| `DATABASE_URL` | Frontend | Database URL passed during Amplify build | No |
| `AMPLIFY_DIFF_DEPLOY` | Frontend | AWS Amplify deployment setting | No |
| `AMPLIFY_MONOREPO_APP_ROOT` | Frontend | AWS Amplify app root path (frontend) | No |
| `NODE_ENV` | Frontend | Node environment mode | No |

## Team

* Mohd Arsh: Implemented AWS Bedrock integration, Amazon Titan text embeddings, Pinecone vector indexing, RAG conversational chat router, dual-layer search and composite ranking services, PostgreSQL database migrations, Pydantic domain models, unit test suites, Next.js 16 frontend porting, Next.js reverse proxy rewrites, GitHub Actions CI/CD and keep-alive workflows, and Render and Amplify deployment configurations.
* Raunak (Coder'Phantom95): Implemented the multi-source data ingestion pipeline, including GitHub, Hugging Face, Kaggle, and OpenRouter data fetchers, normalization and deduplication stages, PostgreSQL ingestion repository, local setup scripts, and EC2 systemd timer configurations.
* Aryan: Implemented EC2 deployment scripts, Gunicorn startup configurations, API gateway routing, Nginx reverse proxy configuration, systemd service definitions, user and category routers, ranking recalculation scripts, and ranking integration tests.
* Vansh: Implemented frontend dashboard components, layout styling, sidebar toggle responsiveness, and mock authentication login interface.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.