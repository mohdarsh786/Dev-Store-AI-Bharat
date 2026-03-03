# DevStore - AI-Powered Developer Marketplace

DevStore is a Google Play Store-inspired marketplace that enables developers to discover, evaluate, and integrate APIs, Models, and Datasets through intelligent, context-aware search powered by AWS services.

## Features

- **Semantic Search**: Natural language search with RAG (Retrieval-Augmented Generation)
- **Smart Ranking**: Composite scoring based on relevance, popularity, optimization, and freshness
- **Glassmorphism UI**: Modern, custom-designed interface with dark/light themes
- **Multilingual Support**: English, Hindi, Hinglish, Tamil, Telugu, Bengali
- **One-Click Boilerplate**: Generate ready-to-use starter code in Python, JavaScript, or TypeScript
- **Health Monitoring**: Real-time API health status and uptime tracking
- **Solution Blueprints**: Visual architecture diagrams showing resource relationships

## Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.11)
- **Deployment**: AWS Lambda + API Gateway
- **Database**: RDS Aurora PostgreSQL
- **Search**: Amazon OpenSearch Service
- **AI/ML**: Amazon Bedrock (Claude 3, Titan Embeddings)
- **Storage**: Amazon S3
- **Authentication**: AWS Cognito

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Vanilla CSS with CSS Modules (NO Tailwind, NO template libraries)
- **Routing**: React Router
- **Visualization**: React Flow
- **i18n**: react-i18next
- **Deployment**: S3 + CloudFront

## Project Structure

```
devstore/
├── backend/              # FastAPI backend
│   ├── main.py          # Application entry point
│   ├── config.py        # Configuration
│   ├── models/          # Data models
│   ├── services/        # Business logic
│   ├── clients/         # AWS clients
│   ├── routers/         # API routes
│   └── tests/           # Test suite
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # UI components
│   │   ├── pages/       # Page components
│   │   ├── services/    # API clients
│   │   ├── styles/      # CSS files
│   │   └── utils/       # Utilities
│   └── public/          # Static assets
└── .kiro/
    └── specs/
        └── devstore/    # Project specifications
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL
- AWS Account

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your AWS credentials and database URL

# Run development server
uvicorn main:app --reload --port 8000
```

API documentation: http://localhost:8000/docs

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env with your API URL

# Run development server
npm run dev
```

App: http://localhost:3000

## Testing

### Backend Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run property-based tests
pytest -m property_test
```

### Frontend Tests

```bash
cd frontend

# Run tests
npm test

# Run with UI
npm run test:ui
```

## Deployment

### Backend (AWS Lambda)

```bash
cd backend

# Package Lambda function
pip install -r requirements.txt -t package/
cd package && zip -r ../lambda.zip .
cd .. && zip -g lambda.zip *.py

# Deploy
aws lambda update-function-code \
  --function-name devstore-api \
  --zip-file fileb://lambda.zip
```

### Frontend (S3 + CloudFront)

```bash
cd frontend

# Build
npm run build

# Deploy to S3
aws s3 sync dist/ s3://devstore-frontend-prod/

# Invalidate CloudFront cache
aws cloudfront create-invalidation \
  --distribution-id YOUR_DIST_ID \
  --paths "/*"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CloudFront (CDN)                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                React Frontend (S3)                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│          API Gateway + Lambda (FastAPI)                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌──────────────────┬──────────────────┬──────────────┐
        ↓                  ↓                  ↓              ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐
│ RDS Aurora   │  │  OpenSearch  │  │   Bedrock    │  │   S3    │
│ PostgreSQL   │  │   (Vector    │  │   (Claude/   │  │ (Assets)│
│              │  │   Search)    │  │   Titan)     │  │         │
└──────────────┘  └──────────────┘  └──────────────┘  └─────────┘
```

## Team

- **Mohd Arsh** - AI/Search (RAG, Ranking, Multilingual)
- **Raunak** - Data/Infrastructure (Crawlers, Database, Health Monitoring)
- **Vansh** - Frontend (UI/UX, Glassmorphism Design)
- **Aryan** - Backend/Deployment (API, Boilerplate Generator, DevOps)

## Hackathon

Built for **AI4 Bharat powered by AWS** hackathon.

## License

MIT
