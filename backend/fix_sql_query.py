# Fix the SQL query to use correct column names

old_query = '''            cursor.execute("""
                SELECT id, name, description, type, url, documentation_url, 
                       pricing_type, github_stars, downloads, last_updated,
                       health_status, source
                FROM resources
                ORDER BY id
            """)'''

new_query = '''            cursor.execute("""
                SELECT id, name, description, type, source_url, documentation_url, 
                       pricing_type, github_stars, download_count, updated_at,
                       health_status, source
                FROM resources
                ORDER BY id
            """)'''

# Read the file
with open('rag/ingestor.py', 'r') as f:
    content = f.read()

# Replace the query
content = content.replace(old_query, new_query)

# Write back
with open('rag/ingestor.py', 'w') as f:
    f.write(content)

print("Fixed SQL query in ingestor.py")
