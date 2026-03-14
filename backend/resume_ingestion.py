"""Resume ingestion for remaining resources"""
import logging
from clients.bedrock import BedrockClient
from clients.opensearch import OpenSearchClient
from rag.ingestor import DataIngestor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get already indexed resource names to skip them
opensearch = OpenSearchClient()
bedrock = BedrockClient()

# Get all indexed documents
logger.info("Fetching already indexed resources...")
response = opensearch._client.search(
    index='devstore-osaka-db',
    body={
        "size": 10000,
        "_source": ["name"],
        "query": {"match_all": {}}
    }
)

indexed_names = set()
for hit in response['hits']['hits']:
    indexed_names.add(hit['_source']['name'])

logger.info(f"Found {len(indexed_names)} already indexed resources")

# Now run ingestion with skip logic
import psycopg2
from urllib.parse import urlparse
from config import settings
from datetime import datetime

db_url = urlparse(settings.database_url)
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
    SELECT id, name, description, type, source_url, documentation_url, 
           pricing_type, github_stars, download_count, updated_at,
           health_status, source
    FROM resources
    ORDER BY id
""")

rows = cursor.fetchall()
total = len(rows)
skipped = 0
indexed = 0
failed = 0

logger.info(f"Processing {total} total resources...")

for row in rows:
    name = row[1] or 'Unknown'
    
    # Skip if already indexed
    if name in indexed_names:
        skipped += 1
        continue
    
    try:
        resource_dict = {
            'id': row[0],
            'name': name,
            'description': row[2] or 'No description available',
            'category': row[3].lower() if row[3] else 'api',
            'source': row[11] or 'github',
            'source_url': row[4] or '',
            'author': 'Unknown',
            'stars': row[7] or 0,
            'downloads': row[8] or 0,
            'license': None,
            'tags': [],
            'metadata': {
                'documentation_url': row[5],
                'pricing_type': row[6],
                'health_status': row[10],
                'last_updated': row[9].isoformat() if row[9] else None
            }
        }
        
        # Create searchable text
        text = f"Name: {resource_dict['name']} | Description: {resource_dict['description']} | Category: {resource_dict['category']} | Source: {resource_dict['source']}"
        
        # Generate embedding
        embedding = bedrock.generate_embedding(text)
        
        # Prepare document
        document = {
            'name': resource_dict['name'],
            'description': resource_dict['description'][:1000],
            'resource_type': resource_dict['category'].upper(),
            'pricing_type': resource_dict['metadata']['pricing_type'] or 'free',
            'source': resource_dict['source'],
            'source_url': resource_dict['source_url'],
            'author': resource_dict['author'],
            'github_stars': resource_dict['stars'],
            'downloads': resource_dict['downloads'],
            'license': resource_dict['license'] or 'Unknown',
            'tags': resource_dict['tags'],
            'last_updated': resource_dict['metadata']['last_updated'] or datetime.utcnow().isoformat(),
            'health_status': resource_dict['metadata']['health_status'] or 'healthy',
            'embedding': embedding,
            'metadata': resource_dict['metadata']
        }
        
        # Index
        opensearch.index_document(document=document, refresh=False)
        indexed += 1
        
        if indexed % 10 == 0:
            logger.info(f"Progress: Indexed {indexed}, Skipped {skipped}, Failed {failed} (Total: {skipped + indexed + failed}/{total})")
        
    except Exception as e:
        logger.error(f"Failed to process {name}: {e}")
        failed += 1

cursor.close()
conn.close()

# Refresh index
logger.info("Refreshing index...")
opensearch._client.indices.refresh(index='devstore-osaka-db')

logger.info(f"\n{'='*60}")
logger.info(f"RESUME COMPLETE")
logger.info(f"{'='*60}")
logger.info(f"Total resources: {total}")
logger.info(f"Already indexed (skipped): {skipped}")
logger.info(f"Newly indexed: {indexed}")
logger.info(f"Failed: {failed}")
logger.info(f"Total now indexed: {skipped + indexed}")
logger.info(f"{'='*60}")
