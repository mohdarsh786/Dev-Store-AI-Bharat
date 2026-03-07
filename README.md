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
- **Deployment**: EC2 + Gunicorn/Uvicorn (`backend/api_gateway.py`)
- **Database**: RDS Aurora PostgreSQL
- **Search**: Amazon OpenSearch Service
- **AI/ML**: Amazon Bedrock (Claude 3, Titan Embeddings)
- **Storage**: Amazon S3
- **Authentication**: AWS Cognito

### Frontend
- **Framework**: Next.js 16 App Router
- **Build Tool**: Next.js build pipeline
- **Styling**: Vanilla CSS with CSS Modules
- **Routing**: File-based routing with Route Handlers
- **Visualization**: React Flow
- **i18n**: react-i18next
- **Deployment**: S3 + CloudFront

## Project Structure

```
devstore/
├── backend/              # FastAPI backend
│   ├── api_gateway.py   # EC2 application entry point
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

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- AWS Account (configured with IAM user)

### 1. Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your AWS credentials

# Run server
uvicorn api_gateway:app --reload --port 8000
```

API docs: http://localhost:8000/api/docs

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
# Create frontend/.env.local with:
# BACKEND_URL=http://localhost:8000

# Run dev server
npm run dev
```

App: http://localhost:3000

### 3. Test Connections

```bash
cd backend/tests
python test_connections_simple.py
```

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - Detailed setup guide
- [AWS_SETUP_GUIDE.md](AWS_SETUP_GUIDE.md) - AWS configuration
- [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) - Database structure
- [EC2_DEPLOYMENT_GUIDE.md](EC2_DEPLOYMENT_GUIDE.md) - Production deployment

## Testing

```bash
# Backend tests
cd backend/tests
python test_connections_simple.py  # Quick connection test
pytest                              # Full test suite

# Frontend checks
cd frontend
npm run build
```

## Deployment

See [EC2_DEPLOYMENT_GUIDE.md](EC2_DEPLOYMENT_GUIDE.md) for production deployment instructions.

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
│          EC2 FastAPI Gateway (Gunicorn/Uvicorn)              │
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
