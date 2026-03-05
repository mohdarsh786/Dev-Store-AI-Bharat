# DevStore - Design Document

## Overview

DevStore is an AI-powered marketplace for developers to discover and integrate APIs, Models, and Datasets. The system combines semantic search using RAG (Retrieval-Augmented Generation) with a ranking algorithm to surface the most relevant resources based on natural language queries.

### Core Design Principles

1. **Serverless-First Architecture**: Leverage AWS Lambda and managed services to minimize operational overhead
2. **Semantic Understanding**: Use vector embeddings and LLMs to understand developer intent beyond keyword matching
3. **Modular Design**: Separate concerns across search, ranking, data ingestion, and presentation layers
4. **Performance**: Target sub-500ms search latency through caching and optimized vector search
5. **Extensibility**: Design for easy addition of new resource types and ranking factors

### Key Technical Decisions

- **FastAPI on Lambda**: Use Mangum adapter to run FastAPI in Lambda for familiar Python web framework with serverless benefits
- **OpenSearch for Vector Search**: Native KNN support with proven scalability for semantic search
- **Amazon Bedrock**: Managed LLM service (Claude) for RAG without model hosting complexity
- **Dual Database Strategy**: PostgreSQL for structured metadata + DocumentDB for flexible API specifications
- **React with Vanilla CSS**: Custom glassmorphism design system without framework dependencies

---

## Architecture

### System Context Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         External Systems                        │
│  - GitHub API  - Hugging Face  - API Providers  - Users         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                          DevStore Platform                     │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Frontend    │  │   Backend    │  │  AI/Search   │          │
│  │  (React)     │→ │  (FastAPI)   │→ │  (Bedrock)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                              ↓                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  PostgreSQL  │  │  OpenSearch  │  │  Background  │          │
│  │  (Metadata)  │  │  (Vectors)   │  │  Workers     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────────────────────────────────────────┘
```

### Component Architecture


#### 1. Frontend Layer (React + S3 + CloudFront)

**Responsibilities**:
- Render glassmorphism UI with custom CSS
- Handle user interactions and state management
- Communicate with backend via REST API
- Implement client-side routing and caching

**Key Components**:
- `SearchInterface`: Natural language search input with filters
- `CategoryBrowser`: Grid view of resources by category
- `ResourceDetail`: Comprehensive resource information page
- `SolutionBlueprint`: Visual architecture diagram generator
- `ThemeProvider`: Dark/light mode with CSS custom properties

**Technology Stack**:
- React 18 with hooks
- React Router for navigation
- CSS Modules for scoped styling
- React Flow for architecture diagrams
- react-i18next for multilingual support

#### 2. API Layer (FastAPI + Lambda + API Gateway)

**Responsibilities**:
- Expose REST endpoints for frontend
- Orchestrate search queries across OpenSearch and PostgreSQL
- Handle authentication and authorization
- Generate boilerplate code
- Track analytics and usage

**Key Endpoints**:
```
GET  /api/v1/resources              # List resources with filters
GET  /api/v1/resources/{id}         # Get resource details
POST /api/v1/search                 # Semantic search
GET  /api/v1/categories             # List categories
POST /api/v1/boilerplate/generate   # Generate starter code
GET  /api/v1/users/profile          # User profile
POST /api/v1/users/track            # Track user actions
GET  /api/v1/health                 # Health check
```

**Technology Stack**:
- FastAPI 0.100+
- Mangum for Lambda integration
- Pydantic for request/response validation
- boto3 for AWS service integration


#### 3. Search & AI Layer (OpenSearch + Bedrock)

**Responsibilities**:
- Generate embeddings for resources and queries
- Perform vector similarity search
- Extract intent from natural language queries
- Generate contextual explanations
- Rank results using hybrid scoring

**Search Pipeline**:
```
User Query → Intent Extraction (Bedrock) → Query Embedding (Bedrock)
    ↓
Vector Search (OpenSearch KNN) → Candidate Resources
    ↓
Hybrid Ranking (Semantic + Popularity + Optimization + Freshness)
    ↓
Filtered & Sorted Results → Frontend
```

**RAG Architecture**:
1. **Indexing Phase**: Generate embeddings for all resources (name + description + tags)
2. **Query Phase**: 
   - Extract intent (API/Model/Dataset needs, tech stack, constraints)
   - Generate query embedding
   - Retrieve top-K similar resources from OpenSearch
   - Re-rank using composite scoring algorithm
   - Return results with explanations

**Technology Stack**:
- Amazon Bedrock (Claude 3 for text, Titan for embeddings)
- OpenSearch 2.x with KNN plugin
- Custom ranking algorithm implementation

#### 4. Data Layer (PostgreSQL + DocumentDB)

**Responsibilities**:
- Store structured resource metadata
- Track user profiles and search history
- Store flexible API specifications
- Maintain ranking scores
- Support transactional operations

**Database Strategy**:
- **PostgreSQL (RDS Aurora)**: Primary datastore for structured data
  - Resources, categories, users, rankings
  - ACID guarantees for critical operations
  - Efficient joins and aggregations
  
- **DocumentDB**: Secondary store for unstructured data
  - Full API specifications (OpenAPI/Swagger docs)
  - Model configuration files
  - Dataset schemas
  - Flexible schema evolution


#### 5. Background Workers (Lambda + EventBridge)

**Responsibilities**:
- Crawl external sources for new resources
- Update popularity metrics (GitHub stars, downloads)
- Perform health checks on APIs
- Recompute ranking scores
- Generate embeddings for new resources

**Worker Functions**:
- `crawler-github`: Fetch repository metadata from GitHub API
- `crawler-huggingface`: Fetch model information from Hugging Face
- `health-checker`: Verify API endpoints are responsive
- `ranking-updater`: Recompute daily ranking scores
- `embedding-generator`: Generate vectors for new resources

**Scheduling**:
- Health checks: Every 6 hours
- Popularity updates: Daily at 2 AM UTC
- Ranking recomputation: Daily at 3 AM UTC
- Crawlers: Continuous with rate limiting

---

## Components and Interfaces

### Search Service

**Purpose**: Orchestrate semantic search across vector and metadata stores

**Interface**:
```python
class SearchService:
    async def search(
        self,
        query: str,
        filters: SearchFilters,
        user_context: Optional[UserContext] = None
    ) -> SearchResults:
        """
        Perform semantic search with ranking.
        
        Args:
            query: Natural language search query
            filters: Pricing, type, category filters
            user_context: User profile for personalization
            
        Returns:
            Ranked list of resources with relevance scores
        """
        pass
    
    async def extract_intent(self, query: str) -> Intent:
        """Extract structured intent from natural language query."""
        pass
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate 1536-dimensional embedding vector."""
        pass
```

**Dependencies**:
- BedrockClient: LLM inference
- OpenSearchClient: Vector search
- PostgreSQLClient: Metadata retrieval


### Ranking Service

**Purpose**: Compute composite scores for resource ranking

**Interface**:
```python
class RankingService:
    def compute_score(
        self,
        resource: Resource,
        query_embedding: List[float],
        user_context: Optional[UserContext] = None
    ) -> RankingScore:
        """
        Compute composite ranking score.
        
        Score = 0.4 * semantic_relevance 
              + 0.3 * popularity 
              + 0.2 * optimization 
              + 0.1 * freshness
        
        Returns:
            RankingScore with component breakdowns
        """
        pass
    
    def compute_semantic_relevance(
        self,
        resource_embedding: List[float],
        query_embedding: List[float]
    ) -> float:
        """Cosine similarity between embeddings."""
        pass
    
    def compute_popularity(self, resource: Resource) -> float:
        """Normalized score from GitHub stars, downloads, users."""
        pass
    
    def compute_optimization(self, resource: Resource) -> float:
        """Score based on latency, cost, documentation quality."""
        pass
    
    def compute_freshness(self, resource: Resource) -> float:
        """Score based on last update and health status."""
        pass
```

**Scoring Algorithm Details**:

1. **Semantic Relevance (40%)**:
   - Cosine similarity between query and resource embeddings
   - Range: [0, 1] where 1 is perfect match
   - Formula: `dot(query_vec, resource_vec) / (norm(query_vec) * norm(resource_vec))`

2. **Popularity (30%)**:
   - Normalized combination of GitHub stars, downloads, active users
   - Formula: `0.4 * norm(github_stars) + 0.4 * norm(downloads) + 0.2 * norm(active_users)`
   - Normalization: Min-max scaling to [0, 1]

3. **Optimization (20%)**:
   - Latency score: `1 - (latency_ms / 5000)` capped at [0, 1]
   - Cost efficiency: `1 / (1 + log10(cost_per_1k_requests))`
   - Documentation quality: Binary (0.5 if exists, 1.0 if comprehensive)
   - Formula: `0.4 * latency + 0.3 * cost + 0.3 * docs`

4. **Freshness (10%)**:
   - Days since last update: `exp(-days_since_update / 180)`
   - Health status: 1.0 (healthy), 0.5 (degraded), 0.0 (down)
   - Formula: `0.6 * recency + 0.4 * health`


### Boilerplate Generator Service

**Purpose**: Generate ready-to-use starter code for selected resources

**Interface**:
```python
class BoilerplateGenerator:
    async def generate(
        self,
        resources: List[Resource],
        language: str,  # 'python', 'javascript', 'typescript'
        options: GenerationOptions
    ) -> BoilerplatePackage:
        """
        Generate boilerplate code package.
        
        Returns:
            BoilerplatePackage with code files, README, .env template
        """
        pass
    
    def generate_integration_code(
        self,
        resource: Resource,
        language: str
    ) -> str:
        """Generate API/SDK integration code."""
        pass
    
    def generate_env_template(
        self,
        resources: List[Resource]
    ) -> str:
        """Generate .env.example with required variables."""
        pass
    
    def generate_readme(
        self,
        resources: List[Resource],
        language: str
    ) -> str:
        """Generate setup instructions."""
        pass
```

**Template Structure**:
```
boilerplate-{timestamp}/
├── README.md                 # Setup instructions
├── .env.example             # Environment variables template
├── requirements.txt         # Python dependencies (if Python)
├── package.json             # Node dependencies (if JS/TS)
├── src/
│   ├── config.py/ts         # Configuration loader
│   ├── clients/             # API client wrappers
│   │   ├── api_name.py/ts
│   │   └── model_name.py/ts
│   ├── utils/               # Helper functions
│   │   └── error_handler.py/ts
│   └── main.py/ts           # Example usage
└── tests/                   # Basic test structure
    └── test_integration.py/ts
```


### Resource Crawler Service

**Purpose**: Discover and index new resources from external sources

**Interface**:
```python
class ResourceCrawler:
    async def crawl_github(
        self,
        query: str,
        resource_type: str
    ) -> List[RawResource]:
        """Crawl GitHub for repositories matching criteria."""
        pass
    
    async def crawl_huggingface(
        self,
        filters: Dict[str, Any]
    ) -> List[RawResource]:
        """Crawl Hugging Face for models and datasets."""
        pass
    
    async def extract_metadata(
        self,
        raw_resource: RawResource
    ) -> Resource:
        """Extract structured metadata from raw data."""
        pass
    
    async def validate_resource(
        self,
        resource: Resource
    ) -> ValidationResult:
        """Validate resource has required fields and working URLs."""
        pass
```

**Crawling Strategy**:
- Rate limiting: 100 requests/hour per source
- Exponential backoff on errors
- Respect robots.txt and API rate limits
- Cache responses for 24 hours
- Prioritize popular resources (stars > 100)

### Health Monitoring Service

**Purpose**: Track API availability and performance

**Interface**:
```python
class HealthMonitor:
    async def check_health(
        self,
        resource: Resource
    ) -> HealthStatus:
        """
        Perform health check on resource.
        
        Returns:
            HealthStatus with status, latency, error details
        """
        pass
    
    async def compute_uptime(
        self,
        resource_id: str,
        days: int = 30
    ) -> float:
        """Calculate uptime percentage over period."""
        pass
    
    async def get_response_time_metrics(
        self,
        resource_id: str
    ) -> ResponseMetrics:
        """Get p50, p95, p99 response times."""
        pass
```

**Health Check Logic**:
- For APIs: Send GET request to health endpoint or base URL
- For Models: Check Hugging Face API status
- For Datasets: Verify download URL is accessible
- Timeout: 10 seconds
- Retry: 3 attempts with exponential backoff
- Status determination:
  - Healthy: Response in < 2s, status 200
  - Degraded: Response in 2-5s or 4xx errors
  - Down: Timeout or 5xx errors


---

## Data Models

### Core Domain Models

#### Resource
```python
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID

class ResourceType(str, Enum):
    API = "api"
    MODEL = "model"
    DATASET = "dataset"

class PricingType(str, Enum):
    FREE = "free"
    PAID = "paid"
    FREEMIUM = "freemium"

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"

class Resource:
    id: UUID
    type: ResourceType
    name: str
    description: str
    long_description: Optional[str]
    pricing_type: PricingType
    price_details: Optional[Dict[str, Any]]
    source_url: str
    documentation_url: Optional[str]
    github_stars: Optional[int]
    download_count: Optional[int]
    active_users: Optional[int]
    health_status: HealthStatus
    last_health_check: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]  # Type-specific fields
    
    # Computed fields
    embedding: Optional[List[float]]  # 1536-dimensional vector
    categories: List[str]
    tags: List[str]
```

#### SearchFilters
```python
class SearchFilters:
    resource_types: Optional[List[ResourceType]]
    pricing_types: Optional[List[PricingType]]
    categories: Optional[List[str]]
    min_stars: Optional[int]
    health_status: Optional[List[HealthStatus]]
    languages: Optional[List[str]]  # Programming languages
```


#### Intent
```python
class Intent:
    """Structured representation of user's search intent."""
    
    primary_need: ResourceType  # What they're primarily looking for
    secondary_needs: List[ResourceType]  # Additional resources needed
    tech_stack: List[str]  # Mentioned technologies (e.g., ['python', 'react'])
    use_case: str  # Extracted use case description
    constraints: Dict[str, Any]  # Pricing, performance, etc.
    language: str  # Query language (en, hi, etc.)
    confidence: float  # Intent extraction confidence [0, 1]
```

#### RankingScore
```python
class RankingScore:
    resource_id: UUID
    semantic_relevance: float  # [0, 1]
    popularity: float  # [0, 1]
    optimization: float  # [0, 1]
    freshness: float  # [0, 1]
    final_score: float  # Weighted combination
    
    # Component breakdowns
    popularity_breakdown: Dict[str, float]  # stars, downloads, users
    optimization_breakdown: Dict[str, float]  # latency, cost, docs
    freshness_breakdown: Dict[str, float]  # recency, health
```

#### SearchResults
```python
class SearchResult:
    resource: Resource
    score: RankingScore
    explanation: str  # Why this was recommended
    snippet: str  # Highlighted relevant text

class SearchResults:
    query: str
    intent: Intent
    results: List[SearchResult]
    total_count: int
    execution_time_ms: float
    filters_applied: SearchFilters
```

#### UserContext
```python
class UserContext:
    user_id: UUID
    preferred_language: str
    tech_stack: List[str]
    search_history: List[str]  # Recent queries
    used_resources: List[UUID]  # Previously used resources
    preferences: Dict[str, Any]
```


#### BoilerplatePackage
```python
class CodeFile:
    path: str  # Relative path in package
    content: str
    language: str

class BoilerplatePackage:
    package_id: str
    resources: List[Resource]
    language: str
    files: List[CodeFile]
    readme: str
    env_template: str
    created_at: datetime
    download_url: str  # S3 presigned URL for ZIP
```

### Database Schema

#### PostgreSQL Tables

**resources**
```sql
CREATE TABLE resources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR(20) NOT NULL CHECK (type IN ('api', 'model', 'dataset')),
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    long_description TEXT,
    pricing_type VARCHAR(20) NOT NULL CHECK (pricing_type IN ('free', 'paid', 'freemium')),
    price_details JSONB,
    source_url VARCHAR(500) NOT NULL,
    documentation_url VARCHAR(500),
    github_stars INTEGER DEFAULT 0,
    download_count INTEGER DEFAULT 0,
    active_users INTEGER DEFAULT 0,
    health_status VARCHAR(20) DEFAULT 'healthy' CHECK (health_status IN ('healthy', 'degraded', 'down')),
    last_health_check TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT valid_urls CHECK (
        source_url ~ '^https?://' AND
        (documentation_url IS NULL OR documentation_url ~ '^https?://')
    )
);

CREATE INDEX idx_resources_type ON resources(type);
CREATE INDEX idx_resources_pricing ON resources(pricing_type);
CREATE INDEX idx_resources_health ON resources(health_status);
CREATE INDEX idx_resources_stars ON resources(github_stars DESC);
CREATE INDEX idx_resources_updated ON resources(updated_at DESC);
```


**categories**
```sql
CREATE TABLE categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,
    parent_id UUID REFERENCES categories(id) ON DELETE CASCADE,
    resource_type VARCHAR(20) CHECK (resource_type IN ('api', 'model', 'dataset', 'all')),
    display_order INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_categories_parent ON categories(parent_id);
CREATE INDEX idx_categories_type ON categories(resource_type);
```

**resource_categories** (many-to-many)
```sql
CREATE TABLE resource_categories (
    resource_id UUID REFERENCES resources(id) ON DELETE CASCADE,
    category_id UUID REFERENCES categories(id) ON DELETE CASCADE,
    PRIMARY KEY (resource_id, category_id)
);

CREATE INDEX idx_rc_resource ON resource_categories(resource_id);
CREATE INDEX idx_rc_category ON resource_categories(category_id);
```

**users**
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cognito_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    preferred_language VARCHAR(10) DEFAULT 'en',
    tech_stack JSONB DEFAULT '[]'::jsonb,
    preferences JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

CREATE INDEX idx_users_cognito ON users(cognito_id);
CREATE INDEX idx_users_email ON users(email);
```

**search_history**
```sql
CREATE TABLE search_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    language VARCHAR(10),
    filters JSONB,
    results_count INTEGER,
    clicked_resources UUID[],
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_search_user ON search_history(user_id);
CREATE INDEX idx_search_created ON search_history(created_at DESC);
```


**resource_usage**
```sql
CREATE TABLE resource_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    resource_id UUID REFERENCES resources(id) ON DELETE CASCADE,
    action VARCHAR(50) NOT NULL CHECK (action IN ('view', 'download_boilerplate', 'test', 'bookmark')),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_usage_user ON resource_usage(user_id);
CREATE INDEX idx_usage_resource ON resource_usage(resource_id);
CREATE INDEX idx_usage_action ON resource_usage(action);
CREATE INDEX idx_usage_created ON resource_usage(created_at DESC);
```

**resource_rankings**
```sql
CREATE TABLE resource_rankings (
    resource_id UUID REFERENCES resources(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    semantic_relevance_avg FLOAT DEFAULT 0,
    popularity_score FLOAT NOT NULL,
    optimization_score FLOAT NOT NULL,
    freshness_score FLOAT NOT NULL,
    final_score FLOAT NOT NULL,
    rank_position INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (resource_id, date)
);

CREATE INDEX idx_rankings_date ON resource_rankings(date DESC);
CREATE INDEX idx_rankings_score ON resource_rankings(final_score DESC);
CREATE INDEX idx_rankings_position ON resource_rankings(rank_position);
```

**health_checks**
```sql
CREATE TABLE health_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id UUID REFERENCES resources(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL CHECK (status IN ('healthy', 'degraded', 'down')),
    response_time_ms INTEGER,
    error_message TEXT,
    checked_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_health_resource ON health_checks(resource_id);
CREATE INDEX idx_health_checked ON health_checks(checked_at DESC);
```


#### OpenSearch Index Mapping

```json
{
  "settings": {
    "index": {
      "knn": true,
      "knn.algo_param.ef_search": 512
    }
  },
  "mappings": {
    "properties": {
      "resource_id": {
        "type": "keyword"
      },
      "name": {
        "type": "text",
        "analyzer": "standard",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "description": {
        "type": "text",
        "analyzer": "standard"
      },
      "type": {
        "type": "keyword"
      },
      "pricing_type": {
        "type": "keyword"
      },
      "tags": {
        "type": "keyword"
      },
      "categories": {
        "type": "keyword"
      },
      "embedding": {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "nmslib",
          "parameters": {
            "ef_construction": 512,
            "m": 16
          }
        }
      },
      "popularity_score": {
        "type": "float"
      },
      "final_score": {
        "type": "float"
      },
      "health_status": {
        "type": "keyword"
      },
      "created_at": {
        "type": "date"
      },
      "updated_at": {
        "type": "date"
      }
    }
  }
}
```

**KNN Search Query Example**:
```json
{
  "size": 20,
  "query": {
    "bool": {
      "must": [
        {
          "knn": {
            "embedding": {
              "vector": [0.123, 0.456, ...],
              "k": 50
            }
          }
        }
      ],
      "filter": [
        {"terms": {"pricing_type": ["free", "freemium"]}},
        {"terms": {"health_status": ["healthy", "degraded"]}}
      ]
    }
  }
}
```


---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

After analyzing all acceptance criteria, I've identified the following testable properties. Some properties have been combined to avoid redundancy:

- Properties 8.2, 8.3, 8.4 (boilerplate components) can be combined into a single comprehensive property about boilerplate completeness
- Properties 4.1, 4.3, 4.4 (resource detail fields) can be combined into a single property about detail view completeness
- Properties 10.2, 10.3, 10.4 (health metrics) are distinct and provide unique validation value

### Search and Filtering Properties

**Property 1: Pricing Filter Enforcement**

*For any* search query with a pricing filter (free, paid, or both), all returned resources must match the specified pricing type(s).

**Validates: Requirements US-003 (3.3)**

**Property 2: Search Result Ranking Order**

*For any* search query, the returned results must be sorted in descending order by their final composite score (semantic_relevance * 0.4 + popularity * 0.3 + optimization * 0.2 + freshness * 0.1).

**Validates: Requirements US-002 (2.4), US-006 (6.1)**

**Property 3: Multi-Type Search Results**

*For any* search query that extracts intent for multiple resource types (e.g., API and Model), the results must include resources of all requested types.

**Validates: Requirements US-002 (2.3)**

**Property 4: Search Results Grouping**

*For any* search results list, resources must be grouped by their type (API, Model, Dataset) in the response structure.

**Validates: Requirements US-002 (2.5)**

**Property 5: Filter State Persistence**

*For any* pricing filter setting, performing multiple consecutive searches must maintain the same filter value across all searches until explicitly changed.

**Validates: Requirements US-003 (3.2)**


### Ranking Algorithm Properties

**Property 6: Composite Score Calculation**

*For any* resource with computed component scores, the final ranking score must equal exactly: (semantic_relevance * 0.4) + (popularity * 0.3) + (optimization * 0.2) + (freshness * 0.1).

**Validates: Requirements US-006 (6.1)**

**Property 7: Score Component Bounds**

*For any* resource ranking computation, all component scores (semantic_relevance, popularity, optimization, freshness) must be in the range [0, 1].

**Validates: Requirements US-006 (6.1)**

### UI Rendering Properties

**Property 8: Resource Card Completeness**

*For any* resource displayed in a card view, the rendered output must contain the resource's name, description, rating, download/usage count, and pricing information.

**Validates: Requirements US-001 (1.4)**

**Property 9: Category Subcategories Presence**

*For any* main category section (API, Model, or Dataset), the section must include all five subcategories: Top Grossing, Top Free, Top Paid, Trending, and New Releases.

**Validates: Requirements US-001 (1.2)**

**Property 10: Resource Detail View Completeness**

*For any* resource detail page, the rendered output must contain: description, documentation link, pricing information, usage statistics, code examples in at least two languages, and integration requirements.

**Validates: Requirements US-004 (4.1, 4.3, 4.4)**

**Property 11: Theme Toggle State Change**

*For any* theme toggle action, the active theme must change from dark to light or light to dark, and the change must be reflected in the CSS custom properties.

**Validates: Requirements US-007.5 (7.5.6)**


### Boilerplate Generation Properties

**Property 12: Boilerplate Generation Success**

*For any* valid resource and supported language (Python, JavaScript, TypeScript), the boilerplate generator must produce a complete package without errors.

**Validates: Requirements US-008 (8.1)**

**Property 13: Boilerplate Package Completeness**

*For any* generated boilerplate package, it must contain: API integration code, authentication setup, error handling, an .env.example file, and a README with setup instructions.

**Validates: Requirements US-008 (8.2, 8.3, 8.4)**

### Multilingual Support Properties

**Property 14: Multilingual Query Processing**

*For any* search query in a supported language (English, Hindi, Hinglish, Tamil, Telugu, Bengali), the search service must process the query and return results without language-related errors.

**Validates: Requirements US-002 (2.1), US-009 (9.2)**

### Health Monitoring Properties

**Property 15: Health Status Computation**

*For any* resource with a health check endpoint, the health monitoring service must compute and assign a status (healthy, degraded, or down) based on response time and HTTP status code.

**Validates: Requirements US-010 (10.1)**

**Property 16: Uptime Calculation Accuracy**

*For any* resource with health check history over N days, the computed uptime percentage must equal (number of healthy checks / total checks) * 100.

**Validates: Requirements US-010 (10.2)**

**Property 17: Response Time Metrics Availability**

*For any* resource with at least one health check, response time metrics (p50, p95, p99) must be computable and available.

**Validates: Requirements US-010 (10.3)**

**Property 18: Health Check Timestamp Presence**

*For any* resource that has undergone a health check, the last_health_check timestamp must be present and not null.

**Validates: Requirements US-010 (10.4)**


### User Profile and Personalization Properties

**Property 19: User Profile Update on Actions**

*For any* user action (search, resource view, boilerplate download), the user's profile must be updated to reflect the action in their history.

**Validates: Requirements US-007 (7.1)**

### Compatibility and Integration Properties

**Property 20: Resource Compatibility Score Computation**

*For any* pair of resources, a compatibility score must be computable based on shared technologies, data format compatibility, and integration patterns.

**Validates: Requirements US-005 (5.3)**

**Property 21: Integration Complexity Score Computation**

*For any* set of selected resources, an integration complexity score must be computable based on the number of resources, compatibility scores, and required dependencies.

**Validates: Requirements US-005 (5.4)**

---

## Error Handling

### Error Categories

#### 1. Client Errors (4xx)

**Invalid Search Query**
- Status: 400 Bad Request
- Cause: Empty query, invalid filters, malformed JSON
- Response: `{"error": "invalid_query", "message": "Search query cannot be empty", "details": {...}}`
- Recovery: Return error to user with specific validation messages

**Resource Not Found**
- Status: 404 Not Found
- Cause: Requested resource ID doesn't exist
- Response: `{"error": "resource_not_found", "message": "Resource with ID {id} not found"}`
- Recovery: Suggest similar resources or return to search

**Unauthorized Access**
- Status: 401 Unauthorized
- Cause: Missing or invalid authentication token
- Response: `{"error": "unauthorized", "message": "Valid authentication required"}`
- Recovery: Redirect to login page

**Rate Limit Exceeded**
- Status: 429 Too Many Requests
- Cause: User exceeded API rate limits
- Response: `{"error": "rate_limit_exceeded", "message": "Too many requests", "retry_after": 60}`
- Recovery: Implement exponential backoff, show user-friendly message


#### 2. Server Errors (5xx)

**OpenSearch Unavailable**
- Status: 503 Service Unavailable
- Cause: OpenSearch cluster is down or unreachable
- Response: `{"error": "search_unavailable", "message": "Search service temporarily unavailable"}`
- Recovery: Fallback to PostgreSQL keyword search, retry with exponential backoff, alert operations team

**Bedrock API Error**
- Status: 502 Bad Gateway
- Cause: Bedrock API timeout or error
- Response: `{"error": "ai_service_error", "message": "AI service temporarily unavailable"}`
- Recovery: Return cached results if available, fallback to keyword search, log error for investigation

**Database Connection Error**
- Status: 503 Service Unavailable
- Cause: PostgreSQL connection pool exhausted or database unreachable
- Response: `{"error": "database_unavailable", "message": "Service temporarily unavailable"}`
- Recovery: Retry with exponential backoff (max 3 attempts), return cached data if available, alert operations

**Embedding Generation Timeout**
- Status: 504 Gateway Timeout
- Cause: Bedrock embedding generation exceeds timeout (10s)
- Response: `{"error": "embedding_timeout", "message": "Request processing timeout"}`
- Recovery: Queue for async processing, return partial results without semantic ranking

#### 3. External Service Errors

**GitHub API Rate Limit**
- Cause: Exceeded GitHub API rate limits during crawling
- Recovery: Implement token rotation, exponential backoff, cache responses for 24 hours

**Health Check Timeout**
- Cause: Resource API doesn't respond within 10 seconds
- Recovery: Mark as "degraded" after 1 timeout, "down" after 3 consecutive timeouts, retry after 1 hour

**Boilerplate Generation Failure**
- Cause: Template not found, invalid resource metadata
- Response: `{"error": "generation_failed", "message": "Unable to generate boilerplate", "details": {...}}`
- Recovery: Log error with resource details, return generic template, notify development team


### Error Handling Patterns

#### Circuit Breaker Pattern

Implement circuit breakers for external services (Bedrock, OpenSearch, GitHub API):

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
            else:
                raise ServiceUnavailableError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise
```

#### Retry with Exponential Backoff

```python
async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
):
    for attempt in range(max_retries):
        try:
            return await func()
        except RetryableError as e:
            if attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            await asyncio.sleep(delay)
```

#### Graceful Degradation

When AI services are unavailable:
1. Fall back to keyword search (PostgreSQL full-text search)
2. Use cached ranking scores from previous day
3. Return results without explanations
4. Display banner: "Search running in limited mode"


### Logging and Monitoring

**Structured Logging Format**:
```json
{
  "timestamp": "2026-03-02T10:30:45.123Z",
  "level": "ERROR",
  "service": "search-api",
  "function": "semantic_search",
  "error_type": "OpenSearchTimeout",
  "message": "OpenSearch query timeout after 5s",
  "context": {
    "query": "virtual court app dataset",
    "user_id": "uuid",
    "request_id": "req-123",
    "filters": {"pricing": "free"}
  },
  "stack_trace": "..."
}
```

**CloudWatch Alarms**:
- API error rate > 5% for 5 minutes
- OpenSearch latency p95 > 1000ms
- Lambda concurrent executions > 80% of limit
- Database connection pool utilization > 90%
- Health check failure rate > 20%

---

## Testing Strategy

### Dual Testing Approach

DevStore requires both unit tests and property-based tests for comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property tests**: Verify universal properties across all inputs using randomized testing

Both approaches are complementary and necessary. Unit tests catch concrete bugs in specific scenarios, while property tests verify general correctness across a wide input space.

### Property-Based Testing

**Library**: Use `hypothesis` for Python backend, `fast-check` for TypeScript frontend

**Configuration**:
- Minimum 100 iterations per property test (due to randomization)
- Each test must reference its design document property
- Tag format: `# Feature: devstore, Property {number}: {property_text}`

**Example Property Test**:
```python
from hypothesis import given, strategies as st
import pytest

# Feature: devstore, Property 1: Pricing Filter Enforcement
@given(
    query=st.text(min_size=1, max_size=100),
    pricing_filter=st.sampled_from(['free', 'paid', 'freemium'])
)
@pytest.mark.property_test
async def test_pricing_filter_enforcement(query, pricing_filter):
    """
    Property 1: For any search query with a pricing filter,
    all returned resources must match the specified pricing type.
    """
    filters = SearchFilters(pricing_types=[pricing_filter])
    results = await search_service.search(query, filters)
    
    for result in results.results:
        assert result.resource.pricing_type == pricing_filter
```


### Unit Testing Strategy

**Test Organization**:
```
tests/
├── unit/
│   ├── test_search_service.py
│   ├── test_ranking_service.py
│   ├── test_boilerplate_generator.py
│   ├── test_health_monitor.py
│   └── test_crawler.py
├── integration/
│   ├── test_search_api.py
│   ├── test_resource_api.py
│   └── test_user_api.py
├── property/
│   ├── test_search_properties.py
│   ├── test_ranking_properties.py
│   ├── test_boilerplate_properties.py
│   └── test_health_properties.py
└── e2e/
    └── test_user_flows.py
```

**Unit Test Focus Areas**:
1. **Edge Cases**:
   - Empty search queries
   - Resources with missing optional fields
   - Extreme ranking score values (0, 1)
   - Health checks with network timeouts

2. **Error Conditions**:
   - Invalid resource IDs
   - Malformed API responses
   - Database connection failures
   - Bedrock API errors

3. **Specific Examples**:
   - Known good search queries with expected results
   - Specific ranking calculations with known inputs/outputs
   - Sample boilerplate generation for each language

**Example Unit Test**:
```python
@pytest.mark.asyncio
async def test_search_with_empty_query_returns_error():
    """Test that empty queries are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        await search_service.search("", SearchFilters())
    
    assert "query cannot be empty" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_ranking_score_with_known_values():
    """Test ranking calculation with specific known values."""
    resource = create_test_resource(
        github_stars=1000,
        download_count=5000,
        health_status="healthy"
    )
    query_embedding = [0.1] * 1536
    
    score = ranking_service.compute_score(resource, query_embedding)
    
    # Verify component weights
    expected = (
        score.semantic_relevance * 0.4 +
        score.popularity * 0.3 +
        score.optimization * 0.2 +
        score.freshness * 0.1
    )
    assert abs(score.final_score - expected) < 0.001
```


### Integration Testing

**Test Scenarios**:
1. **End-to-End Search Flow**:
   - User submits query → Intent extraction → Vector search → Ranking → Results
   - Verify all components work together
   - Test with real OpenSearch and PostgreSQL instances

2. **Boilerplate Generation Flow**:
   - Select resources → Generate code → Create ZIP → Upload to S3
   - Verify generated code is syntactically valid
   - Test all supported languages

3. **Health Monitoring Flow**:
   - Scheduled check → API call → Status update → Database write
   - Verify health status propagates to search results

**Example Integration Test**:
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_integration_flow(test_db, test_opensearch):
    """Test complete search flow with real dependencies."""
    # Setup: Insert test resources
    resource = await test_db.insert_resource({
        "name": "Test API",
        "type": "api",
        "pricing_type": "free",
        "description": "A test API for virtual courts"
    })
    
    # Generate and index embedding
    embedding = await bedrock_client.generate_embedding(
        f"{resource.name} {resource.description}"
    )
    await test_opensearch.index_resource(resource.id, embedding)
    
    # Execute search
    results = await search_service.search(
        "virtual court API",
        SearchFilters(pricing_types=["free"])
    )
    
    # Verify
    assert len(results.results) > 0
    assert results.results[0].resource.id == resource.id
    assert results.results[0].score.final_score > 0
```

### Performance Testing

**Load Testing Targets**:
- Search API: 100 requests/second with p95 latency < 500ms
- Resource detail API: 200 requests/second with p95 latency < 200ms
- Boilerplate generation: 10 requests/second with p95 latency < 3s

**Tools**: Use Locust or k6 for load testing

**Example Load Test**:
```python
from locust import HttpUser, task, between

class DevStoreUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def search(self):
        self.client.post("/api/v1/search", json={
            "query": "machine learning API",
            "filters": {"pricing_types": ["free"]}
        })
    
    @task(1)
    def get_resource_detail(self):
        self.client.get(f"/api/v1/resources/{self.resource_id}")
```

### Test Coverage Goals

- Unit test coverage: > 80% for business logic
- Integration test coverage: All critical user flows
- Property test coverage: All 21 correctness properties
- E2E test coverage: Top 5 user journeys

---

## Deployment Architecture

### AWS Infrastructure

**Frontend (S3 + CloudFront)**:
```
S3 Bucket: devstore-frontend-prod
├── index.html
├── static/
│   ├── js/
│   ├── css/
│   └── assets/

CloudFront Distribution:
- Origin: S3 bucket
- Cache behavior: Cache static assets for 1 year
- Custom domain: devstore.example.com
- SSL: ACM certificate
```

**Backend (Lambda + API Gateway)**:
```
API Gateway: devstore-api-prod
├── /api/v1/search (POST) → search-lambda
├── /api/v1/resources (GET) → resources-lambda
├── /api/v1/resources/{id} (GET) → resource-detail-lambda
├── /api/v1/boilerplate/generate (POST) → boilerplate-lambda
└── /api/v1/users/* → user-lambda

Lambda Functions:
- Runtime: Python 3.11
- Memory: 1024 MB (search), 512 MB (others)
- Timeout: 30s (search), 10s (others)
- Concurrency: Reserved 100 for search
```

**Databases**:
```
RDS Aurora PostgreSQL:
- Instance: db.r6g.large (2 vCPU, 16 GB RAM)
- Multi-AZ: Yes
- Backup: Daily snapshots, 7-day retention
- Connection pooling: RDS Proxy

OpenSearch:
- Instance: 3x r6g.large.search
- Storage: 500 GB EBS per node
- Replicas: 2 per index
- Snapshots: Hourly to S3
```

### CI/CD Pipeline

**Frontend Pipeline** (GitHub Actions → S3):
```yaml
name: Deploy Frontend
on:
  push:
    branches: [main]
    paths: ['frontend/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build
        run: |
          cd frontend
          npm install
          npm run build
      - name: Deploy to S3
        run: |
          aws s3 sync frontend/build/ s3://devstore-frontend-prod/
          aws cloudfront create-invalidation --distribution-id $DIST_ID --paths "/*"
```

**Backend Pipeline** (GitHub Actions → Lambda):
```yaml
name: Deploy Backend
on:
  push:
    branches: [main]
    paths: ['backend/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Package Lambda
        run: |
          cd backend
          pip install -r requirements.txt -t package/
          cd package && zip -r ../lambda.zip .
          cd .. && zip -g lambda.zip *.py
      - name: Deploy to Lambda
        run: |
          aws lambda update-function-code \
            --function-name search-lambda \
            --zip-file fileb://lambda.zip
```

### Monitoring and Observability

**CloudWatch Dashboards**:
- API request rates and latencies
- Lambda invocation counts and errors
- Database connection pool metrics
- OpenSearch cluster health
- Cache hit rates

**X-Ray Tracing**:
- Enable for all Lambda functions
- Trace search flow: API Gateway → Lambda → OpenSearch → PostgreSQL
- Identify bottlenecks and optimize

**Alarms**:
- API error rate > 5%
- Search latency p95 > 500ms
- Database CPU > 80%
- Lambda throttling > 10 invocations/minute

---

## Security Considerations

### Authentication and Authorization

- **User Authentication**: AWS Cognito with OAuth 2.0
- **API Authentication**: JWT tokens in Authorization header
- **Service-to-Service**: IAM roles for Lambda → RDS, Lambda → OpenSearch

### Data Protection

- **Encryption at Rest**: 
  - RDS: AES-256 encryption
  - S3: SSE-S3 encryption
  - OpenSearch: Encryption enabled
  
- **Encryption in Transit**:
  - HTTPS only (TLS 1.2+)
  - CloudFront with ACM certificate
  - RDS connections with SSL

### API Security

- **Rate Limiting**: 100 requests/minute per user (API Gateway)
- **Input Validation**: Pydantic models for all requests
- **SQL Injection Prevention**: Parameterized queries only
- **XSS Prevention**: Content Security Policy headers
- **CORS**: Whitelist frontend domain only

### Secrets Management

- **AWS Secrets Manager**: Store database credentials, API keys
- **Environment Variables**: Lambda environment variables for non-sensitive config
- **Rotation**: Automatic rotation for database passwords (90 days)

---

## Scalability Considerations

### Horizontal Scaling

- **Lambda**: Auto-scales to 1000 concurrent executions
- **RDS**: Read replicas for read-heavy workloads
- **OpenSearch**: Add data nodes as index size grows
- **CloudFront**: Global edge locations for low latency

### Caching Strategy

- **CloudFront**: Cache static assets (1 year TTL)
- **API Gateway**: Cache search results (5 minute TTL)
- **Application**: Redis/ElastiCache for ranking scores (1 hour TTL)
- **Database**: Query result caching in PostgreSQL

### Cost Optimization

- **Lambda**: Use ARM64 (Graviton2) for 20% cost savings
- **RDS**: Use Aurora Serverless v2 for variable workloads
- **OpenSearch**: Use UltraWarm for older data
- **S3**: Use Intelligent-Tiering for boilerplate templates

---

**Document Status**: Complete
**Last Updated**: 2026-03-02
**Next Step**: Create implementation tasks (tasks.md)
