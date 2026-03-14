import sys
import os

# Set up paths to find your modules
sys.path.append(os.path.join(os.getcwd(), 'backend'))
sys.path.append(os.path.join(os.getcwd(), 'backend/rag'))

try:
    # We found your models are in backend/rag/models.py
    from rag.models import Resource
    # We still need to find where your 'SessionLocal' or 'engine' is. 
    # Since find didn't show a simple 'db.py', it's likely in a 'database' folder or __init__
    # Let's try to import the db session from the common locations:
    try:
        from clients.db import SessionLocal
    except ImportError:
        try:
            from database import SessionLocal
        except ImportError:
            print("❌ Could not find SessionLocal. Please run 'ls backend' so I can find your DB client.")
            sys.exit(1)

    db = SessionLocal()
    r = db.query(Resource).first()
    
    if r:
        print("\n✅ DATA STRUCTURE FOUND:")
        data = {c.name: getattr(r, c.name) for c in r.__table__.columns}
        for key, value in data.items():
            print(f"{key}: {type(value).__name__} (Example: {value})")
    else:
        print("⚠️ Database connected, but Resource table is empty.")

except Exception as e:
    print(f"❌ Error during inspection: {e}")
