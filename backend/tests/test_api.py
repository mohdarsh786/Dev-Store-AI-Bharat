"""Test the FastAPI application"""
import sys
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    """Test health endpoint"""
    response = client.get("/api/v1/health")
    print(f"Health Check: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    return True

def test_search():
    """Test search endpoint"""
    response = client.post("/api/v1/search", json={
        "query": "machine learning API",
        "limit": 5
    })
    print(f"\nSearch Test: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    return True

def test_intent_search():
    """Test intent search endpoint"""
    response = client.post("/api/v1/search/intent", json={
        "query": "I need a free NLP model",
        "limit": 5
    })
    print(f"\nIntent Search Test: {response.status_code}")
    print(f"Response keys: {list(response.json().keys())}")
    assert response.status_code == 200
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("FastAPI Application Tests")
    print("=" * 60)
    
    try:
        test_health()
        test_search()
        test_intent_search()
        
        print("\n" + "=" * 60)
        print("✅ All API tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
