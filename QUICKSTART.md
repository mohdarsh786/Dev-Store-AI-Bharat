# DevStore - Quick Start Guide

## Minimal Working Prototype

This is a minimal working prototype of DevStore with core functionality:
- ✅ Bedrock client for AI operations (text generation + embeddings)
- ✅ **Full Bedrock-powered search** with intent extraction and vector search
- ✅ Ranking service with composite scoring
- ✅ Basic backend API with search and resources endpoints
- ✅ Glassmorphism frontend with theme toggle
- ✅ Search interface with mock data
- ✅ Resource cards display

## Bedrock Search Flow

The search endpoint now implements the full AI-powered flow:

1. **Intent Extraction** - Uses Bedrock (Claude 3) to analyze the query and extract:
   - Resource types (API, Model, Dataset)
   - Pricing preference (free, paid, both)
   - Key terms

2. **Embedding Generation** - Uses Bedrock (Titan Embeddings) to convert query to vector

3. **Vector Search** - Performs KNN search in OpenSearch using the embedding

4. **Ranking** - Computes composite scores using:
   - Semantic relevance (40%)
   - Popularity (30%)
   - Optimization (20%)
   - Freshness (10%)

5. **Grouping** - Groups results by resource type for organized display

## Running the Application

### Backend

```bash
cd backend

# Install dependencies (if not already done)
pip install -r requirements.txt

# Run the development server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/v1/health`

### Frontend

```bash
cd frontend

# Install dependencies (if not already done)
npm install

# Run the development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Features Implemented

### Backend
- **Bedrock Client** (`backend/clients/bedrock.py`)
  - Text generation with Claude 3
  - Embedding generation with Titan
  - Circuit breaker pattern
  - Retry logic with exponential backoff

- **API Endpoints**
  - `POST /api/v1/search` - Search resources
  - `GET /api/v1/resources` - List resources
  - `GET /api/v1/resources/{id}` - Get resource details
  - `GET /api/v1/health` - Health check

- **Ranking Service** (`backend/services/ranking.py`)
  - Semantic relevance scoring
  - Popularity scoring
  - Optimization scoring
  - Freshness scoring
  - Weighted composite scoring

### Frontend
- **Glassmorphism Design**
  - Glass cards with backdrop blur
  - Glass buttons with hover effects
  - Glass inputs with focus states
  - Smooth transitions (300ms)

- **Theme Toggle**
  - Light/Dark mode support
  - LocalStorage persistence
  - Smooth color transitions

- **Search Interface**
  - Natural language search bar
  - Real-time results display
  - Resource cards with type badges
  - Pricing indicators (Free/Paid)

- **Components**
  - `SearchBar` - Search input with submit
  - `ResourceCard` - Display resource info
  - `ThemeToggle` - Switch between themes
  - `HomePage` - Main landing page

## Next Steps

To continue development:

1. **Connect Backend to Frontend**
   - Update `HomePage.jsx` to call real API endpoints
   - Add API base URL configuration

2. **Implement Real Search**
   - Connect Bedrock client to search endpoint
   - Integrate OpenSearch for vector search
   - Add database queries for resources

3. **Add More Pages**
   - Resource detail page
   - Category browsing
   - User profile

4. **Deploy**
   - Set up AWS infrastructure
   - Configure environment variables
   - Deploy backend to Lambda
   - Deploy frontend to S3/CloudFront

## Environment Variables

Create `.env` files in both backend and frontend directories:

### Backend `.env`
```
DATABASE_URL=postgresql://user:pass@localhost:5432/devstore
OPENSEARCH_HOST=localhost
AWS_REGION=us-east-1
S3_BUCKET_BOILERPLATE=devstore-boilerplate
S3_BUCKET_CRAWLER_DATA=devstore-crawler
```

### Frontend `.env`
```
VITE_API_URL=http://localhost:8000
```

## Notes

- Currently using mock data for search results
- AWS services (Bedrock, OpenSearch, RDS) need to be configured
- Authentication not yet implemented
- Tests not included in this minimal prototype
