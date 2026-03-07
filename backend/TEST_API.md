# 🚀 Test Your API - Quick Guide

## ✅ What's Ready

Your API is now connected to the ingestion data! Here's what you can do:

## 1. Start the API Server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

## 2. Test the Endpoints

### Get Statistics
```bash
curl http://localhost:8000/api/resources/stats
```

**Response:**
```json
{
  "total_resources": 2500,
  "models": 1200,
  "datasets": 600,
  "repositories": 700,
  "by_source": {
    "huggingface": 1100,
    "openrouter": 100,
    "github": 700,
    "kaggle": 600
  }
}
```

### Search Resources
```bash
curl "http://localhost:8000/api/resources/search?q=bert&limit=5"
```

**Response:**
```json
{
  "query": "bert",
  "total": 45,
  "limit": 5,
  "offset": 0,
  "results": [
    {
      "name": "bert-base-uncased",
      "description": "BERT base model",
      "source": "huggingface",
      "category": "model",
      "stars": 1234,
      "downloads": 5678900,
      ...
    }
  ]
}
```

### Get Trending Resources
```bash
curl "http://localhost:8000/api/resources/trending?limit=10"
```

### Filter by Category
```bash
curl "http://localhost:8000/api/resources/search?q=python&category=model"
```

### Filter by Source
```bash
curl "http://localhost:8000/api/resources/search?q=api&source=github"
```

### Get Categories
```bash
curl http://localhost:8000/api/resources/categories
```

**Response:**
```json
{
  "categories": [
    {"name": "model", "count": 1200},
    {"name": "dataset", "count": 600},
    {"name": "solution", "count": 500},
    {"name": "api", "count": 200}
  ]
}
```

### Get Sources
```bash
curl http://localhost:8000/api/resources/sources
```

### Refresh Cache (After Running Ingestion)
```bash
curl -X POST http://localhost:8000/api/resources/refresh
```

## 3. Test in Browser

Open your browser and visit:

- **API Docs**: http://localhost:8000/docs
- **Stats**: http://localhost:8000/api/resources/stats
- **Search**: http://localhost:8000/api/resources/search?q=python
- **Trending**: http://localhost:8000/api/resources/trending

## 4. Connect Your Frontend

Update your frontend to use these endpoints:

```javascript
// In your frontend (e.g., React)
const API_BASE = 'http://localhost:8000';

// Search resources
async function searchResources(query) {
  const response = await fetch(
    `${API_BASE}/api/resources/search?q=${encodeURIComponent(query)}&limit=20`
  );
  return response.json();
}

// Get trending
async function getTrending(category = null) {
  const url = category 
    ? `${API_BASE}/api/resources/trending?category=${category}`
    : `${API_BASE}/api/resources/trending`;
  const response = await fetch(url);
  return response.json();
}

// Get stats
async function getStats() {
  const response = await fetch(`${API_BASE}/api/resources/stats`);
  return response.json();
}
```

## 5. Update Data

When you want to refresh the data:

```bash
# Run ingestion
cd backend
python ingestion/test_orchestrator.py

# Refresh API cache
curl -X POST http://localhost:8000/api/resources/refresh
```

## Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/resources/search` | GET | Search resources by query |
| `/api/resources/trending` | GET | Get trending resources |
| `/api/resources/categories` | GET | Get all categories with counts |
| `/api/resources/sources` | GET | Get all sources with counts |
| `/api/resources/stats` | GET | Get overall statistics |
| `/api/resources/refresh` | POST | Refresh data cache |

## Query Parameters

### `/api/resources/search`
- `q` (required): Search query
- `category` (optional): Filter by category (model, dataset, api, solution)
- `source` (optional): Filter by source (huggingface, openrouter, github, kaggle)
- `limit` (optional): Results per page (default: 20, max: 100)
- `offset` (optional): Pagination offset (default: 0)

### `/api/resources/trending`
- `category` (optional): Filter by category
- `limit` (optional): Number of results (default: 20, max: 100)

## Example Frontend Integration

```jsx
// React component example
import { useState, useEffect } from 'react';

function ResourceSearch() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `http://localhost:8000/api/resources/search?q=${encodeURIComponent(query)}`
      );
      const data = await response.json();
      setResults(data.results);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search resources..."
      />
      <button onClick={handleSearch} disabled={loading}>
        {loading ? 'Searching...' : 'Search'}
      </button>
      
      <div>
        {results.map((resource) => (
          <div key={resource.source_url}>
            <h3>{resource.name}</h3>
            <p>{resource.description}</p>
            <p>⭐ {resource.stars} | 📥 {resource.downloads}</p>
            <a href={resource.source_url} target="_blank">View</a>
          </div>
        ))}
      </div>
    </div>
  );
}
```

## Next Steps

1. ✅ **Start API**: `uvicorn main:app --reload`
2. ✅ **Test endpoints**: Use curl or browser
3. ✅ **Connect frontend**: Update API calls
4. ✅ **Schedule ingestion**: Run `test_orchestrator.py` daily
5. ✅ **Monitor**: Check `/api/resources/stats`

## Troubleshooting

### "File not found" error
Make sure ingestion has run:
```bash
python ingestion/test_orchestrator.py
```

### Empty results
Check if JSON files exist:
```bash
ls backend/ingestion/output/
```

### Cache not updating
Refresh the cache:
```bash
curl -X POST http://localhost:8000/api/resources/refresh
```

---

**Your API is now serving real data from HuggingFace, OpenRouter, GitHub, and Kaggle!** 🎉
