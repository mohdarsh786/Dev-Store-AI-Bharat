# Fix the database connection in ingest_from_database method

new_connection_code = '''        try:
            # Connect to database
            logger.info("Connecting to PostgreSQL database...")
            
            # Parse database URL from settings
            from urllib.parse import urlparse
            db_url = urlparse(settings.database_url)
            
            conn = psycopg2.connect(
                host=db_url.hostname,
                port=db_url.port or 5432,
                database=db_url.path[1:],  # Remove leading /
                user=db_url.username,
                password=db_url.password,
                sslmode='require'
            )'''

old_connection_code = '''        try:
            # Connect to database
            logger.info("Connecting to PostgreSQL database...")
            conn = psycopg2.connect(
                host=settings.db_host,
                port=settings.db_port,
                database=settings.db_name,
                user=settings.db_user,
                password=settings.db_password
            )'''

# Read the file
with open('rag/ingestor.py', 'r') as f:
    content = f.read()

# Replace the connection code
content = content.replace(old_connection_code, new_connection_code)

# Write back
with open('rag/ingestor.py', 'w') as f:
    f.write(content)

print("Fixed database connection in ingestor.py")
