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

## 🚀 Quick Start

**Your system is ready to run!** All dependencies are installed and 2,240 real resources are available.

### One-Command Start

```bash
start_dev.bat
```

This will:
1. Start backend on http://localhost:8000
2. Start frontend on http://localhost:3000
3. Open browser automatically

**That's it!** 🎉

### Manual Start (Alternative)

**Terminal 1 - Backend:**
```bash
cd backend
.venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Then open: http://localhost:3000

### Verify Setup

```bash
verify_setup.bat
```

### First Time Setup

If you haven't set up yet:
```bash
quick_setup.bat
```

## Documentation

### Quick Start Guides
- **[START_HERE.md](START_HERE.md)** - 🚀 Start here! Quick start in 3 steps
- **[READY_TO_RUN.md](READY_TO_RUN.md)** - ✅ Verification that system is ready
- **[SETUP_CHECKLIST.md](SETUP_CHECKLIST.md)** - 📋 Complete setup checklist
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - 🎯 Getting started guide

### Detailed Guides
- [COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md) - Full setup guide (all phases)
- [CURRENT_STATUS.md](CURRENT_STATUS.md) - Project status and features
- [AWS_SETUP_GUIDE.md](AWS_SETUP_GUIDE.md) - AWS configuration
- [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) - Database structure
- [EC2_DEPLOYMENT_GUIDE.md](EC2_DEPLOYMENT_GUIDE.md) - Production deployment

### Backend Documentation
- `backend/ingestion/PRODUCTION_PIPELINE.md` - Ingestion pipeline guide
- `backend/ingestion/QUICKSTART.md` - Ingestion quick start
- `backend/README.md` - Backend overview

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
