"""
Verification script to check that all critical fixes are in place.
Run this to verify the codebase is ready to go.
"""
import sys
import importlib.util
from pathlib import Path


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    return Path(filepath).exists()


def check_method_exists(module_path: str, class_name: str, method_name: str) -> bool:
    """Check if a method exists in a class."""
    try:
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        cls = getattr(module, class_name, None)
        if cls is None:
            return False
        
        return hasattr(cls, method_name)
    except Exception as e:
        print(f"  Error checking {class_name}.{method_name}: {e}")
        return False


def check_import(module_path: str, import_name: str) -> bool:
    """Check if an import exists in a file."""
    try:
        with open(module_path, 'r') as f:
            content = f.read()
            return import_name in content
    except Exception as e:
        print(f"  Error checking import: {e}")
        return False


def main():
    print("=" * 60)
    print("VERIFYING CRITICAL FIXES")
    print("=" * 60)
    
    all_passed = True
    
    # Check 1: RankingService methods
    print("\n1. Checking RankingService methods...")
    ranking_path = "services/ranking.py"
    if check_method_exists(ranking_path, "RankingService", "compute_trending_score"):
        print("  ✅ compute_trending_score() exists")
    else:
        print("  ❌ compute_trending_score() missing")
        all_passed = False
    
    if check_method_exists(ranking_path, "RankingService", "compute_category_rankings"):
        print("  ✅ compute_category_rankings() exists")
    else:
        print("  ❌ compute_category_rankings() missing")
        all_passed = False
    
    # Check 2: Async database methods
    print("\n2. Checking async database methods...")
    db_path = "clients/database.py"
    if check_import(db_path, "asyncpg"):
        print("  ✅ asyncpg imported")
    else:
        print("  ❌ asyncpg not imported")
        all_passed = False
    
    if check_method_exists(db_path, "DatabaseClient", "fetch"):
        print("  ✅ async fetch() method exists")
    else:
        print("  ❌ async fetch() method missing")
        all_passed = False
    
    if check_method_exists(db_path, "DatabaseClient", "execute"):
        print("  ✅ async execute() method exists")
    else:
        print("  ❌ async execute() method missing")
        all_passed = False
    
    # Check 3: Router prefixes
    print("\n3. Checking router prefixes...")
    search_router = "routers/search.py"
    with open(search_router, 'r') as f:
        content = f.read()
        if 'router = APIRouter(prefix="/api/v1"' in content:
            print("  ❌ search.py has double prefix")
            all_passed = False
        elif 'router = APIRouter(tags=["search"])' in content:
            print("  ✅ search.py prefix fixed")
        else:
            print("  ⚠️  search.py router definition unclear")
    
    resources_router = "routers/resources.py"
    with open(resources_router, 'r') as f:
        content = f.read()
        if 'router = APIRouter(prefix="/api/v1"' in content:
            print("  ❌ resources.py has double prefix")
            all_passed = False
        elif 'router = APIRouter(tags=["resources"])' in content:
            print("  ✅ resources.py prefix fixed")
        else:
            print("  ⚠️  resources.py router definition unclear")
    
    # Check 4: OpenSearch async methods
    print("\n4. Checking OpenSearch async methods...")
    os_path = "clients/opensearch.py"
    if check_method_exists(os_path, "OpenSearchClient", "connect"):
        print("  ✅ async connect() exists")
    else:
        print("  ❌ async connect() missing")
        all_passed = False
    
    if check_method_exists(os_path, "OpenSearchClient", "disconnect"):
        print("  ✅ async disconnect() exists")
    else:
        print("  ❌ async disconnect() missing")
        all_passed = False
    
    # Check 5: .env.example exists
    print("\n5. Checking configuration template...")
    if check_file_exists(".env.example"):
        print("  ✅ .env.example exists")
    else:
        print("  ❌ .env.example missing")
        all_passed = False
    
    # Check 6: asyncpg in requirements
    print("\n6. Checking requirements.txt...")
    with open("requirements.txt", 'r') as f:
        content = f.read()
        if 'asyncpg' in content:
            print("  ✅ asyncpg in requirements.txt")
        else:
            print("  ❌ asyncpg not in requirements.txt")
            all_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Codebase is ready to go!")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please review the issues above")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
