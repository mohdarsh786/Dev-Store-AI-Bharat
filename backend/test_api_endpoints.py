"""
Test API endpoints to verify they work correctly
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(name, url):
    """Test an endpoint and print results"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ {name}: {response.status_code}")
            print(f"  Response keys: {list(data.keys())}")
            return True
        else:
            print(f"✗ {name}: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False

def main():
    print("Testing API Endpoints")
    print("=" * 70)
    
    tests = [
        ("Root", f"{BASE_URL}/"),
        ("Health", f"{BASE_URL}/api/v1/health"),
        ("Stats", f"{BASE_URL}/api/resources/stats"),
        ("Categories", f"{BASE_URL}/api/resources/categories"),
        ("Sources", f"{BASE_URL}/api/resources/sources"),
        ("Search", f"{BASE_URL}/api/resources/search?q=gpt&limit=3"),
        ("Trending", f"{BASE_URL}/api/resources/trending?limit=3"),
    ]
    
    passed = 0
    failed = 0
    
    for name, url in tests:
        if test_endpoint(name, url):
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ {failed} tests failed")

if __name__ == "__main__":
    main()
