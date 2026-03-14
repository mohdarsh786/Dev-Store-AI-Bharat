import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_schema_details(url, db_name):
    try:
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        # Query to get column details, data types, and nullability
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'resources'
            ORDER BY column_name;
        """
        cur.execute(query)
        schema = cur.fetchall()
        cur.close()
        conn.close()
        return {row[0]: {"type": row[1], "null": row[2]} for row in schema}
    except Exception as e:
        print(f"❌ Error connecting to {db_name}: {e}")
        return None

def compare():
    print("🔍 Comparing Schemas: RDS vs Neon...")
    
    rds_schema = get_schema_details(os.getenv("OLD_DATABASE_URL"), "RDS (Old)")
    neon_schema = get_schema_details(os.getenv("DATABASE_URL"), "Neon (New)")

    if not rds_schema or not neon_schema:
        print("🛑 Comparison failed due to connection issues.")
        return

    all_cols = set(rds_schema.keys()) | set(neon_schema.keys())
    
    print("\n" + "="*60)
    print(f"{'Column Name':<20} | {'RDS Type':<15} | {'Neon Type':<15} | {'Match?'}")
    print("-" * 60)

    mismatches = 0
    for col in sorted(all_cols):
        rds_info = rds_schema.get(col, {"type": "MISSING", "null": ""})
        neon_info = neon_schema.get(col, {"type": "MISSING", "null": ""})
        
        match = "✅" if rds_info == neon_info else "❌"
        if match == "❌": mismatches += 1
        
        print(f"{col:<20} | {rds_info['type']:<15} | {neon_info['type']:<15} | {match}")

    print("="*60)
    if mismatches == 0:
        print("🔥 PERFECT! Schema is an exact mirror copy.")
    else:
        print(f"⚠️ Found {mismatches} mismatches! Check data types carefully.")

if __name__ == "__main__":
    compare()
