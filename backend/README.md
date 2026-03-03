# DevStore Backend

FastAPI-based backend for the DevStore AI-powered developer marketplace.

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL
- AWS Account with access to:
  - RDS Aurora
  - OpenSearch Service
  - Amazon Bedrock
  - S3

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running Locally

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run property-based tests only
pytest -m property_test
```

## Project Structure

```
backend/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration management
├── models/              # Pydantic data models
├── services/            # Business logic services
├── clients/             # AWS and external service clients
├── routers/             # API route handlers
├── tests/               # Test suite
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── property/       # Property-based tests
└── requirements.txt     # Python dependencies
```

## Deployment

### Lambda Deployment

1. Package the application:
```bash
pip install -r requirements.txt -t package/
cd package && zip -r ../lambda.zip .
cd .. && zip -g lambda.zip *.py
```

2. Deploy to AWS Lambda:
```bash
aws lambda update-function-code \
  --function-name devstore-api \
  --zip-file fileb://lambda.zip
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /api/v1/health` - Health check
- `POST /api/v1/search` - Semantic search
- `GET /api/v1/resources` - List resources
- `GET /api/v1/resources/{id}` - Get resource details
- `GET /api/v1/categories` - List categories
- `POST /api/v1/boilerplate/generate` - Generate boilerplate code
- `GET /api/v1/users/profile` - Get user profile
- `POST /api/v1/users/track` - Track user actions

## License

MIT
