"""Test intent search endpoint with real Bedrock call"""
import requests
import json

def test_intent_search():
    """Test the /search/intent endpoint"""
    url = "http://localhost:8000/api/v1/search/intent"
    
    payload = {
        "query": "I need a free machine learning model for text classification",
        "limit": 5
    }
    
    print("=" * 60)
    print("Testing Intent Search Endpoint")
    print("=" * 60)
    print(f"\nURL: {url}")
    print(f"Query: {payload['query']}")
    print("\nSending request...")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Success!")
            print(f"\nResponse:")
            print(f"  - Query: {data.get('query')}")
            print(f"  - Total Results: {data.get('total')}")
            print(f"  - Source: {data.get('source')}")
            print(f"  - Intent: {data.get('intent')}")
            
            if data.get('results'):
                print(f"\n  Results:")
                for i, result in enumerate(data['results'][:3], 1):
                    print(f"    {i}. {result.get('name')} ({result.get('resource_type')})")
            else:
                print(f"\n  ⚠️ No results returned (index might be empty)")
            
            # Check if it's using AWS or mock
            if data.get('source') == 'aws':
                print(f"\n✅ Using AWS Bedrock + OpenSearch!")
            else:
                print(f"\n⚠️ Using mock data (AWS not configured or failed)")
                
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to backend")
        print("Make sure backend is running: uvicorn main:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_intent_search()
