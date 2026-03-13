import psycopg2
from urllib.parse import urlparse

database_url = "postgresql://devstore_admin:J.nMDc5dRCc:2kW96lV.53$qPxKb@devstore-postgres.cbauqomywavz.ap-northeast-3.rds.amazonaws.com:5432/postgres?sslmode=require"
db_url = urlparse(database_url)

conn = psycopg2.connect(
    host=db_url.hostname,
    port=db_url.port or 5432,
    database=db_url.path[1:],
    user=db_url.username,
    password=db_url.password,
    sslmode='require'
)

cursor = conn.cursor()
cursor.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'resources'
    ORDER BY ordinal_position
""")

print("Columns in resources table:")
for row in cursor.fetchall():
    print(f"  - {row[0]}")

cursor.close()
conn.close()
