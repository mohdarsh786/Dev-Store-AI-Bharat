import os
import psycopg2
from pinecone import Pinecone
from opensearchpy import OpenSearch
from dotenv import load_dotenv

load_dotenv()

def audit():
    # Initializing with default values to avoid KeyError
    report = {
        'neon_count': 0, 'rds_count': 0, 
        'pinecone_count': 0, 'opensearch_count': 0,
        'neon_cols': set(), 'rds_cols': set()
    }
    
    print("\n⏳ Auditing all systems... Please wait.")

    # --- 1. Neon Connection ---
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM resources")
        report['neon_count'] = cur.fetchone()[0]
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'resources'")
        report['neon_cols'] = set(row[0] for row in cur.fetchall())
        conn.close()
        print("✅ Neon SQL: Connected")
    except Exception as e:
        print(f"❌ Neon SQL Error: {e}")

    # --- 2. RDS Connection ---
    try:
        url = os.getenv("OLD_DATABASE_URL")
        if url:
            conn = psycopg2.connect(url)
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM resources")
            report['rds_count'] = cur.fetchone()[0]
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'resources'")
            report['rds_cols'] = set(row[0] for row in cur.fetchall())
            conn.close()
            print("✅ RDS SQL: Connected")
        else:
            print("⚠️ RDS SQL: OLD_DATABASE_URL not found in .env")
    except Exception as e:
        print(f"❌ RDS SQL Error: {e}\n   (Check if endpoint ends with .amazonaws.com)")

    # --- 3. Pinecone Connection ---
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        idx = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        report['pinecone_count'] = idx.describe_index_stats()['total_vector_count']
        print("✅ Pinecone: Connected")
    except Exception as e:
        print(f"❌ Pinecone Error: {e}")

    # --- 4. OpenSearch Connection ---
    try:
        host = os.getenv("OPENSEARCH_HOST", "").replace("https://", "")
        if host:
            os_client = OpenSearch(hosts=[{'host': host, 'port': 443}], use_ssl=True)
            report['opensearch_count'] = os_client.count(index=os.getenv("OPENSEARCH_INDEX_NAME"))['count']
            print("✅ OpenSearch: Connected")
    except Exception:
        print("❌ OpenSearch: Authorization Failed (403), skipping count.")

    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print("📊 MIGRATION AUDIT SUMMARY")
    print("="*50)
    print(f"RDS (Old DB):      {report['rds_count']} records")
    print(f"Neon (New DB):     {report['neon_count']} records")
    print(f"Pinecone (New):    {report['pinecone_count']} vectors")
    print(f"OpenSearch (Old):  {report['opensearch_count']} vectors")
    
    if report['rds_cols'] and report['neon_cols']:
        missing = report['rds_cols'] - report['neon_cols']
        print(f"Schema:            {'✅ Match' if not missing else f'⚠️ Missing: {missing}'}")
    
    print("-"*50)
    # Decisions
    if report['neon_count'] > 0 and report['neon_count'] == report['pinecone_count']:
        print("🟢 VERDICT: Sync is 100% complete!")
    else:
        gap = abs(report['neon_count'] - report['pinecone_count'])
        print(f"🔴 VERDICT: {gap} records missing in Pinecone. DO NOT delete RDS yet.")
    print("="*50 + "\n")

if __name__ == "__main__":
    audit()
